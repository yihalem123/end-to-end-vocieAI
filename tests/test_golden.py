"""Golden replays: frozen real extractions run through verify + score.

The raw_extraction in each golden is a real gpt-5.6-terra response against the
Phase-2 utterance-id schema, captured once and frozen — so these tests exercise
verification and scoring against genuine model output, deterministically and
offline. Regenerate via the extraction probe if the plan or schema changes.
"""
import json
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.extract import verify_and_coerce
from server.postcall.score import score_call

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_plan(ROOT / "plans" / "icu_nurse.yaml")


def replay(name: str):
    golden = json.loads((ROOT / "tests" / "golden" / f"{name}.json").read_text("utf-8"))
    verified = verify_and_coerce(golden["raw_extraction"], golden["entries"], PLAN)
    return verified, score_call(PLAN, verified)


def test_cooperative_candidate() -> None:
    verified, result = replay("cooperative")
    assert all(e.verified for e in verified.values())      # every quote traced
    assert verified["icu_years"].value == 6.5
    assert verified["shift_availability"].value is True    # typed bool, no substrings
    assert verified["certifications"].value == ["BLS", "ACLS"]  # CCRN pending excluded
    assert result.knocked_out is None
    assert result.needs_review is False
    assert result.score is not None and result.score >= 0.85
    assert result.scoring_version == "icu-nurse-v2"


def test_knockout_rambler() -> None:
    verified, result = replay("knockout_rambler")
    assert verified["rn_license_active"].value is False    # correction understood
    assert verified["rn_license_active"].verified is True  # utterance-traced...
    assert result.knocked_out == "rn_license_active"       # ...so it MAY disqualify
    assert result.score == 0.0
    assert result.needs_review is True
    assert verified["icu_years"].contradictory is True     # "twelve... eleven" caught
    assert any("contradictory" in r for r in result.reasons)
