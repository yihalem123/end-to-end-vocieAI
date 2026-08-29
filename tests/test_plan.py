"""Plan loading and InterviewState: recording, advancement, knockouts."""
from pathlib import Path

import pytest

from server.engine.plan import InterviewState, load_plan

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"


def make_state() -> InterviewState:
    return InterviewState(load_plan(PLAN_PATH))


def test_load_real_plan() -> None:
    plan = load_plan(PLAN_PATH)
    assert plan.scoring_version == "icu-nurse-v2"
    assert [s.field for s in plan.steps][:3] == ["consent", "rn_license_state",
                                                "rn_license_active"]
    assert plan.steps[0].knockout == {"equals": False}
    assert abs(sum(plan.weights.values()) - 1.0) < 1e-9


def test_load_rejects_unknown_field_in_weights(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "persona: p\nconsent: c\nsteps:\n  - field: a\n    type: str\n"
        "weights: {nonexistent: 1.0}\nscoring_version: v1\n"
    )
    with pytest.raises(ValueError, match="nonexistent"):
        load_plan(bad)


def test_float_coercion_tolerates_surrounding_words() -> None:
    # Golden-run regression: extraction returned "six and a half years" -> the
    # value died in float(). Digits embedded in prose must still parse; pure
    # word-numbers still fail (schema now asks for digits).
    st = make_state()
    assert st.record("icu_years", "6.5 years", quote="q") is True
    assert st.fields["icu_years"].value == 6.5
    assert st.record("icu_years", "about 3 years or so", quote="q") is True
    assert st.fields["icu_years"].value == 3.0


def test_record_coerces_types() -> None:
    st = make_state()
    st.record("consent", True, quote="sure, go ahead")
    st.record("icu_years", "5", quote="five years")       # str -> float
    st.record("certifications", "BLS, ACLS", quote="BLS and ACLS")  # str -> list
    assert st.fields["consent"].value is True
    assert st.fields["icu_years"].value == 5.0
    assert st.fields["certifications"].value == ["BLS", "ACLS"]


def test_record_unknown_field_is_rejected() -> None:
    st = make_state()
    assert st.record("favorite_color", "blue", quote="blue") is False
    assert "favorite_color" not in st.fields


def test_advance_requires_current_field_recorded() -> None:
    # LLM-signaled advancement, engine-validated: the signal is refused until
    # the current step's field is actually recorded.
    st = make_state()
    assert st.current_step.field == "consent"
    assert st.request_advance() is False                  # nothing recorded yet
    st.record("consent", True, quote="yes")
    assert st.request_advance() is True
    assert st.current_step.field == "rn_license_state"


def test_advance_skips_steps_already_filled() -> None:
    # Volunteered info: "License is active in Texas" fills two fields at once.
    st = make_state()
    st.record("consent", True, quote="yes")
    st.record("rn_license_state", "Texas", quote="active in Texas")
    st.record("rn_license_active", True, quote="active in Texas")
    assert st.request_advance() is True
    assert st.current_step.field == "icu_years"           # skipped two filled steps


def test_knockout_detected_on_record() -> None:
    st = make_state()
    st.record("consent", True, quote="yes")
    st.request_advance()
    st.record("rn_license_state", "Ohio", quote="Ohio")
    st.record("rn_license_active", False, quote="it lapsed")
    assert st.knocked_out is None  # live LLM capture is provisional, never adverse


def test_ambiguous_boolean_is_nullable_not_false() -> None:
    st = make_state()
    assert st.record("consent", "maybe", quote="maybe") is False
    assert "consent" not in st.fields
    assert st.record("consent", "not sure", quote="not sure") is False
    assert "consent" not in st.fields


def test_done_only_after_all_steps() -> None:
    st = make_state()
    answers = {
        "consent": True, "rn_license_state": "Texas", "rn_license_active": True,
        "icu_years": 5, "certifications": ["CCRN"], "shift_availability": True,
        "earliest_start": "two weeks", "pay_expectation": "55/hr",
    }
    for field, value in answers.items():
        assert not st.done
        st.record(field, value, quote=str(value))
        st.request_advance()
    assert st.done


def test_next_needed_ignores_cursor_lag() -> None:
    # The model may record answers without ever signaling advance_step; the
    # prompt must still point at the first genuinely unanswered step.
    st = make_state()
    st.record("consent", True, quote="yes")
    st.record("rn_license_state", "Texas", quote="Texas")
    assert st.current_step.field == "consent"          # cursor never moved
    assert st.next_needed is not None
    assert st.next_needed.field == "rn_license_active"  # but the need is clear


def test_next_needed_none_when_all_filled() -> None:
    st = make_state()
    fill = {"bool": True, "float": 1.0, "list": ["x"], "str": "x"}
    for s in st.plan.steps:
        assert st.record(s.field, fill[s.type], quote="q")
    assert st.next_needed is None


def test_next_askable_skips_askless_steps() -> None:
    # rn_license_active has no ask text: it rides along with the license
    # question, so it can never be the spoken objective itself.
    st = make_state()
    st.record("consent", True, quote="yes")
    st.record("rn_license_state", "Texas", quote="Texas")
    assert st.next_needed.field == "rn_license_active"
    assert st.next_askable.field == "icu_years"


def test_history_is_bounded_window() -> None:
    st = make_state()
    for i in range(30):
        st.add_history("user" if i % 2 else "assistant", f"line {i}")
    assert len(st.recent_history(8)) == 8
    assert st.recent_history(8)[-1]["content"] == "line 29"
