"""Extraction pure parts: schema from plan, transcript render, quote verification."""
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.extract import build_schema, render_transcript, verify_and_coerce

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"
PLAN = load_plan(PLAN_PATH)

ENTRIES = [
    {"role": "agent", "text": "Is now a good time?"},
    {"role": "caller", "text": "Yeah sure, now works.", "utterance_id": "u1"},
    {"role": "agent", "text": "How many years of ICU experience do you have?"},
    {"role": "caller", "text": "About six and a half years, mostly nights.",
     "utterance_id": "u2"},
    {"role": "agent", "text": "Which certifications do you hold?", "interrupted": True},
    {"role": "caller", "text": "BLS and ACLS.", "utterance_id": "u3"},
]
TRANSCRIPT = render_transcript(ENTRIES)


def test_render_transcript_tags_speakers_and_interruptions() -> None:
    assert "CALLER [u1]: Yeah sure, now works." in TRANSCRIPT
    assert "AGENT: Which certifications do you hold? [cut off by caller]" in TRANSCRIPT


def test_schema_is_strict_and_covers_every_plan_field() -> None:
    schema = build_schema(PLAN)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {s.field for s in PLAN.steps}
    per_field = schema["properties"]["icu_years"]
    assert set(per_field["required"]) == {
        "value", "quote", "utterance_id", "confidence", "contradictory"}
    assert per_field["properties"]["value"]["type"] == ["string", "null"]


def test_schema_descriptions_are_type_aware() -> None:
    # Golden-run regression: the model wrote "six and a half" for a float field.
    schema = build_schema(PLAN)
    assert "digits" in schema["properties"]["icu_years"]["properties"]["value"]["description"]
    assert "true or false" in schema["properties"]["consent"]["properties"]["value"]["description"]


def _raw(field: str, value, quote, confidence=0.9,
         utterance_id="u2", contradictory=False) -> dict:
    return {field: {"value": value, "quote": quote, "confidence": confidence,
                    "utterance_id": utterance_id,
                    "contradictory": contradictory}}


def test_verified_quote_keeps_confidence_and_coerces() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "About six and a half years"), ENTRIES, PLAN)
    assert out["icu_years"].value == 6.5
    assert out["icu_years"].confidence == 0.9
    assert out["icu_years"].verified is True


def test_verification_is_punctuation_and_case_insensitive() -> None:
    out = verify_and_coerce(
        _raw("consent", "true", "yeah sure NOW works!!!", utterance_id="u1"),
        ENTRIES, PLAN)
    assert out["consent"].confidence == 0.9


def test_fabricated_quote_zeroes_confidence() -> None:
    # The hard rule: evidence that isn't in the transcript is worthless.
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "I have ten years of experience"), ENTRIES, PLAN)
    assert out["icu_years"].value == 6.5
    assert out["icu_years"].confidence == 0.0


def test_missing_quote_zeroes_confidence() -> None:
    out = verify_and_coerce(_raw("icu_years", "6.5", None), ENTRIES, PLAN)
    assert out["icu_years"].confidence == 0.0


def test_null_value_means_field_absent() -> None:
    out = verify_and_coerce(_raw("pay_expectation", None, None), ENTRIES, PLAN)
    assert "pay_expectation" not in out


def test_uncoercible_value_zeroes_confidence() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "several", "About six and a half years"), ENTRIES, PLAN)
    assert out["icu_years"].value is None
    assert out["icu_years"].confidence == 0.0


def test_agent_words_cannot_verify_candidate_evidence() -> None:
    out = verify_and_coerce(
        _raw("pay_expectation", "55", "How many years of ICU experience",
             utterance_id="u2"),
        ENTRIES, PLAN)
    assert out["pay_expectation"].verified is False
    assert out["pay_expectation"].confidence == 0.0


def test_quote_must_belong_to_named_caller_utterance() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "About six and a half years",
             utterance_id="u1"),
        ENTRIES, PLAN)
    assert out["icu_years"].verified is False


def test_contradiction_flag_survives_verification_for_scoring_gate() -> None:
    out = verify_and_coerce(
        _raw("icu_years", "6.5", "About six and a half years",
             contradictory=True),
        ENTRIES, PLAN)
    assert out["icu_years"].verified is True
    assert out["icu_years"].contradictory is True


def test_ambiguous_boolean_value_stays_unknown_not_false() -> None:
    out = verify_and_coerce(
        _raw("consent", "maybe", "Yeah sure, now works", utterance_id="u1"),
        ENTRIES, PLAN)
    assert out["consent"].value is None
    assert out["consent"].confidence == 0.0
