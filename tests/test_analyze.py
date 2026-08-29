"""Configured advisory analyses: schema from plan, shaping, report inclusion."""
from pathlib import Path

from server.engine.plan import load_plan
from server.postcall.analyze import build_analysis_schema, shape_analyses

PLAN = load_plan(Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml")


def test_schema_is_strict_and_covers_every_configured_analysis() -> None:
    schema = build_analysis_schema(PLAN)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"summary", "followups", "interviewer_notes"}
    assert schema["properties"]["summary"]["description"].startswith("Summarize")


def test_shape_preserves_plan_order_and_titles() -> None:
    shaped = shape_analyses(PLAN, {
        "interviewer_notes": "No hesitations observed.",
        "summary": "Candidate confirmed license and experience.",
        "followups": "1. Ask about CCRN timeline.",
    })
    assert [s["id"] for s in shaped] == ["summary", "followups",
                                        "interviewer_notes"]
    assert shaped[0]["title"] == "Call summary"


def test_shape_drops_empty_answers_instead_of_rendering_blanks() -> None:
    shaped = shape_analyses(PLAN, {"summary": "  ", "followups": "F.",
                                   "interviewer_notes": ""})
    assert [s["id"] for s in shaped] == ["followups"]
