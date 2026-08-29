"""Report assembly, lifecycle completion, and isolated report endpoints."""
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

import server.postcall.report as report_module
from server.app import create_app
from server.config import Settings
from server.postcall.extract import Extracted
from server.postcall.report import (
    build_report, render_html, reports, store_terminal_report)
from server.postcall.score import ScoreResult
from server.realtime.session import SessionLifecycle, SessionStatus

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"


def sample_report() -> dict:
    extracted = {
        "icu_years": Extracted(value=6.5, quote="six and a half", confidence=0.99,
                               utterance_id="u2", verified=True),
        "consent": Extracted(value=True, quote="sure", confidence=0.9,
                             utterance_id="u1", verified=True),
    }
    result = ScoreResult(
        score=0.9, needs_review=False, knocked_out=None,
        scoring_version="icu-nurse-v2", reasons=[],
        subscores={"icu_years": {"subscore": 1.0, "weight": 0.4,
                                 "weighted": 0.4, "confidence": 0.99}},
    )
    conversation = [
        {"role": "agent", "text": "How many years?"},
        {"role": "caller", "text": "Six and a half."},
    ]

    class Turn:
        endpoint_delay = 0.295

    return build_report("abc123", conversation, [Turn()], extracted, result)


def test_build_report_shape() -> None:
    report = sample_report()
    assert report["call_id"] == "abc123"
    assert report["session_state"] == "completed"
    assert report["score"] == 0.9
    assert report["fields"]["icu_years"]["value"] == 6.5
    assert report["endpoint_delays_ms"] == [295]
    assert report["conversation"][1]["text"] == "Six and a half."


def test_render_html_includes_verdict_fields_and_transcript() -> None:
    html = render_html(sample_report())
    assert "score 0.90" in html
    assert "icu_years" in html and "6.5" in html
    assert "Six and a half." in html
    assert "<script" not in html  # escaped output only


def test_report_endpoints() -> None:
    app = create_app(Settings(_env_file=None))
    client = TestClient(app)
    assert client.get("/report/nope").status_code == 404
    assert client.get("/report/nope/view").status_code == 404
    reports["abc123"] = sample_report()
    try:
        assert client.get("/calls").status_code == 404
        assert client.get("/report/abc123").json()["score"] == 0.9
        view = client.get("/report/abc123/view")
        assert view.status_code == 200
        assert "Six and a half." in view.text
    finally:
        del reports["abc123"]


def test_consent_refusal_report_has_no_candidate_analysis() -> None:
    store_terminal_report(
        "refused1",
        [{"role": "caller", "text": "No", "utterance_id": "refused1:u1"}],
        SessionStatus.CONSENT_REFUSED,
        "candidate declined consent; no analysis performed",
    )
    try:
        report = reports["refused1"]
        assert report["session_state"] == "consent_refused"
        assert report["score"] is None
        assert report["fields"] == {}
        assert report["knocked_out"] is None
    finally:
        del reports["refused1"]


def test_postcall_completion_keeps_same_call_id_and_lifecycle(monkeypatch) -> None:
    async def fake_extract(_settings, _plan, _conversation):
        return {}

    monkeypatch.setattr(report_module, "extract_call", fake_extract)
    lifecycle = SessionLifecycle("stable-call")
    lifecycle.transition(SessionStatus.AWAITING_CONSENT)
    lifecycle.transition(SessionStatus.INTERVIEWING)
    lifecycle.transition(SessionStatus.POST_PROCESSING)
    settings = Settings(_env_file=None, plan_path=str(PLAN_PATH))
    asyncio.run(report_module.run_postcall(
        "stable-call", [], [], settings, lifecycle=lifecycle))
    try:
        assert lifecycle.status == SessionStatus.COMPLETED
        assert reports["stable-call"]["call_id"] == "stable-call"
        assert reports["stable-call"]["session_state"] == "completed"
        assert reports["stable-call"]["score"] is None
    finally:
        del reports["stable-call"]


def test_browser_report_polling_never_uses_global_newest_call() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))
    source = client.get("/audio.js").text
    assert 'fetch("/calls")' not in source
    assert "/report/${encodeURIComponent(sessionCallId)}" in source
