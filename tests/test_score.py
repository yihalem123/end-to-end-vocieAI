"""Deterministic scorer: rubric subscores, knockouts, needs_review."""
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.extract import Extracted
from server.postcall.score import score_call

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"


def fields(**kwargs) -> dict:
    """Extracted fields with default confidence 0.9 unless overridden."""
    out = {}
    for name, value in kwargs.items():
        conf = 0.9
        if isinstance(value, tuple):
            value, conf = value
        out[name] = Extracted(value=value, quote=f"said {name}", confidence=conf)
    return out


def full_pass() -> dict:
    return fields(
        consent=True, rn_license_state="Texas", rn_license_active=True,
        icu_years=5.0, certifications=["BLS", "ACLS", "CCRN"],
        shift_availability="nights are fine", earliest_start="two weeks",
        pay_expectation="55 an hour",
    )


def test_strong_candidate_scores_full() -> None:
    result = score_call(load_plan(PLAN_PATH), full_pass())
    assert result.score == 1.0
    assert result.needs_review is False
    assert result.knocked_out is None
    assert result.scoring_version == "icu-nurse-v1"


def test_knockout_zeroes_everything() -> None:
    extracted = full_pass()
    extracted["rn_license_active"] = Extracted(value=False, quote="it lapsed",
                                               confidence=0.95)
    result = score_call(load_plan(PLAN_PATH), extracted)
    assert result.score == 0.0
    assert result.knocked_out == "rn_license_active"


def test_min_full_is_linear_below_threshold() -> None:
    extracted = full_pass()
    extracted["icu_years"] = Extracted(value=1.0, quote="one year", confidence=0.9)
    result = score_call(load_plan(PLAN_PATH), extracted)
    # icu_years subscore 0.5 on weight 0.4: total = 1.0 - 0.4 * 0.5
    assert abs(result.score - 0.8) < 1e-9
    assert result.subscores["icu_years"]["subscore"] == 0.5


def test_expected_list_scores_fraction() -> None:
    extracted = full_pass()
    extracted["certifications"] = Extracted(value=["BLS"], quote="just BLS",
                                            confidence=0.9)
    result = score_call(load_plan(PLAN_PATH), extracted)
    assert abs(result.subscores["certifications"]["subscore"] - 1 / 3) < 1e-9


def test_contains_any_matches_case_insensitively() -> None:
    extracted = full_pass()
    extracted["shift_availability"] = Extracted(value="I can do Night shifts",
                                                quote="night shifts", confidence=0.9)
    result = score_call(load_plan(PLAN_PATH), extracted)
    assert result.subscores["shift_availability"]["subscore"] == 1.0


def test_low_confidence_flags_needs_review() -> None:
    extracted = full_pass()
    extracted["icu_years"] = Extracted(value=5.0, quote="???", confidence=0.2)
    result = score_call(load_plan(PLAN_PATH), extracted)
    assert result.needs_review is True
    assert any("icu_years" in r for r in result.reasons)


def test_missing_weighted_field_scores_zero_and_flags() -> None:
    extracted = full_pass()
    del extracted["earliest_start"]
    result = score_call(load_plan(PLAN_PATH), extracted)
    assert result.subscores["earliest_start"]["subscore"] == 0.0
    assert result.needs_review is True
