"""Extraction pure parts: schema from plan, transcript render, quote verification."""
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.extract import build_schema, render_transcript, verify_and_coerce

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"
PLAN = load_plan(PLAN_PATH)

TRANSCRIPT = render_transcript([
    {"role": "agent", "text": "Is now a good time?"},
    {"role": "caller", "text": "Yeah sure, now works."},
    {"role": "agent", "text": "How many years of ICU experience do you have?"},
    {"role": "caller", "text": "About six and a half years, mostly nights."},
    {"role": "agent", "text": "Which certifications do you hold?", "interrupted": True},
    {"role": "caller", "text": "BLS and ACLS."},
])


def test_render_transcript_tags_speakers_and_interruptions() -> None:
    assert "CALLER: Yeah sure, now works." in TRANSCRIPT
    assert "AGENT: Which certifications do you hold? [cut off by caller]" in TRANSCRIPT


def test_schema_is_strict_and_covers_every_plan_field() -> None:
    schema = build_schema(PLAN)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {s.field for s in PLAN.steps}
    per_field = schema["properties"]["icu_years"]
    assert set(per_field["required"]) == {"value", "quote", "confidence"}
    assert per_field["properties"]["value"]["type"] == ["string", "null"]


def test_schema_descriptions_are_type_aware() -> None:
    # Golden-run regression: the model wrote "six and a half" for a float field.
    schema = build_schema(PLAN)
    assert "digits" in schema["properties"]["icu_years"]["properties"]["value"]["description"]
    assert "true or false" in schema["properties"]["consent"]["properties"]["value"]["description"]


def _raw(field: str, value, quote, confidence=0.9) -> dict:
    return {field: {"value": value, "quote": quote, "confidence": confidence}}


def test_verified_quote_keeps_confidence_and_coerces() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "About six and a half years"), TRANSCRIPT, PLAN)
    assert out["icu_years"].value == 6.5
    assert out["icu_years"].confidence == 0.9


def test_verification_is_punctuation_and_case_insensitive() -> None:
    out = verify_and_coerce(
        _raw("consent", "true", "yeah sure NOW works!!!"), TRANSCRIPT, PLAN)
    assert out["consent"].confidence == 0.9


def test_fabricated_quote_zeroes_confidence() -> None:
    # The hard rule: evidence that isn't in the transcript is worthless.
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "I have ten years of experience"), TRANSCRIPT, PLAN)
    assert out["icu_years"].value == 6.5
    assert out["icu_years"].confidence == 0.0


def test_missing_quote_zeroes_confidence() -> None:
    out = verify_and_coerce(_raw("icu_years", "6.5", None), TRANSCRIPT, PLAN)
    assert out["icu_years"].confidence == 0.0


def test_null_value_means_field_absent() -> None:
    out = verify_and_coerce(_raw("pay_expectation", None, None), TRANSCRIPT, PLAN)
    assert "pay_expectation" not in out


def test_uncoercible_value_zeroes_confidence() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "several", "About six and a half years"), TRANSCRIPT, PLAN)
    assert out["icu_years"].value is None
    assert out["icu_years"].confidence == 0.0
