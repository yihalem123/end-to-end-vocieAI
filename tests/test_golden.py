"""Golden replays: frozen real extractions run through verify + score.

The raw_extraction in each golden file is a real gpt-5.6-terra response,
captured once by scripts and frozen — so these tests exercise the verification
and scoring layers against genuine model output, deterministically and offline.
Regenerate via the extraction probe if the plan schema changes.
"""
import json
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.extract import render_transcript, verify_and_coerce
from server.postcall.score import score_call

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_plan(ROOT / "plans" / "icu_nurse.yaml")


def replay(name: str):
    golden = json.loads((ROOT / "tests" / "golden" / f"{name}.json").read_text("utf-8"))
    verified = verify_and_coerce(golden["raw_extraction"], golden["entries"], PLAN)
    return verified, score_call(PLAN, verified)


def test_cooperative_candidate() -> None:
    verified, result = replay("cooperative")
    assert verified["icu_years"].value == 6.5
    assert verified["consent"].value is True
    # Frozen pre-Phase-2 output has no caller utterance ids and is quarantined.
    assert all(not e.verified for e in verified.values())
    assert result.knocked_out is None
    assert result.needs_review is True
    assert result.score is None
    assert result.scoring_version == "icu-nurse-v2"


def test_knockout_rambler() -> None:
    verified, result = replay("knockout_rambler")
    assert verified["rn_license_active"].value is False   # correction understood
    assert result.knocked_out is None  # no utterance id => cannot disqualify
    assert result.score is None
    assert result.needs_review is True
    assert any("evidence not verified" in r for r in result.reasons)
