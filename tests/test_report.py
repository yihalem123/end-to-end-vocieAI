"""Report assembly, HTML rendering, and the report endpoints."""
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.postcall.extract import Extracted
from server.postcall.report import build_report, render_html, reports
from server.postcall.score import ScoreResult


def sample_report() -> dict:
    extracted = {
        "icu_years": Extracted(value=6.5, quote="six and a half", confidence=0.99),
        "consent": Extracted(value=True, quote="sure", confidence=0.9),
    }
    result = ScoreResult(
        score=0.9, needs_review=False, knocked_out=None,
        scoring_version="icu-nurse-v1", reasons=[],
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
        listing = client.get("/calls").json()
        assert any(c["call_id"] == "abc123" and c["score"] == 0.9 for c in listing)
        assert client.get("/report/abc123").json()["score"] == 0.9
        view = client.get("/report/abc123/view")
        assert view.status_code == 200
        assert "Six and a half." in view.text
    finally:
        del reports["abc123"]
