"""Sentence chunker, Responses-API event assembly, prompt rendering."""
from pathlib import Path

from server.engine.plan import InterviewState, load_plan
import pytest

from server.engine.turn import (
    EngineStreamError,
    SentenceChunker,
    StreamAssembler,
    build_system_prompt,
)

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"


# --- sentence chunker ---

def test_chunker_emits_on_sentence_boundary() -> None:
    ch = SentenceChunker()
    out = list(ch.push("Thanks for calling. How are"))
    out += list(ch.push(" you today? I have"))
    assert out == ["Thanks for calling.", "How are you today?"]
    assert ch.flush() == "I have"


def test_chunker_abbreviation_guard() -> None:
    # A trailing "." is held (could be "Dr." awaiting "Smith") — the final
    # sentence arrives via flush(), which is how the speak path drains it.
    ch = SentenceChunker()
    out = list(ch.push("Dr. Smith works at St. Mary's hospital. Great. "))
    assert out == ["Dr. Smith works at St. Mary's hospital.", "Great."]


def test_chunker_decimal_guard() -> None:
    ch = SentenceChunker()
    out = list(ch.push("You mentioned 4.5 years of experience. Noted."))
    assert out == ["You mentioned 4.5 years of experience."]
    assert ch.flush() == "Noted."


def test_chunker_trailing_boundary_held_until_flush() -> None:
    ch = SentenceChunker()
    assert list(ch.push("Complete sentence.")) == []  # ambiguous: could be "Dr."
    assert ch.flush() == "Complete sentence."
    assert ch.flush() is None


# --- streamed tool-call + text assembly (Responses API typed events) ---

def _events_for_tool_call() -> list[dict]:
    return [
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                  "name": "record_answer", "arguments": ""}},
        {"type": "response.function_call_arguments.delta",
         "item_id": "fc_1", "delta": '{"field": "icu_'},
        {"type": "response.function_call_arguments.delta",
         "item_id": "fc_1", "delta": 'years", "value": 5, "quote": "five years"}'},
        {"type": "response.function_call_arguments.done",
         "item_id": "fc_1",
         "arguments": '{"field": "icu_years", "value": 5, "quote": "five years"}'},
    ]


def test_assembler_builds_tool_call_from_deltas() -> None:
    asm = StreamAssembler()
    for ev in _events_for_tool_call():
        asm.feed(ev)
    assert len(asm.tool_calls) == 1
    call = asm.tool_calls[0]
    assert call.name == "record_answer"
    assert call.arguments == {"field": "icu_years", "value": 5, "quote": "five years"}


def test_assembler_collects_text_deltas() -> None:
    asm = StreamAssembler()
    texts = []
    texts += asm.feed({"type": "response.output_text.delta", "delta": "Thanks! "})
    texts += asm.feed({"type": "response.output_text.delta", "delta": "Got it."})
    assert texts == ["Thanks! ", "Got it."]


def test_assembler_ignores_unknown_events() -> None:
    asm = StreamAssembler()
    assert asm.feed({"type": "response.created"}) == []
    assert asm.feed({"type": "response.completed"}) == []
    assert asm.tool_calls == []


def test_assembler_raises_on_streamed_error_events() -> None:
    # The API reports failures as events INSIDE a 200 stream (found live:
    # credit_balance_exhausted came back as {'type': 'error', ...} and the
    # engine returned an empty reply as if nothing happened). Loud, not silent.
    asm = StreamAssembler()
    with pytest.raises(EngineStreamError, match="no credits"):
        asm.feed({"type": "error",
                  "error": {"code": "credit_balance_exhausted",
                            "message": "You have no credits remaining."}})
    with pytest.raises(EngineStreamError):
        asm.feed({"type": "response.failed",
                  "response": {"error": {"message": "boom"}}})


def test_assembler_malformed_arguments_kept_as_error() -> None:
    asm = StreamAssembler()
    asm.feed({"type": "response.output_item.added",
              "item": {"type": "function_call", "id": "fc_2", "call_id": "c2",
                       "name": "record_answer", "arguments": ""}})
    asm.feed({"type": "response.function_call_arguments.done",
              "item_id": "fc_2", "arguments": "not json"})
    assert asm.tool_calls[0].arguments is None  # surfaced, not crashed


# --- prompt rendering ---

def test_system_prompt_contains_step_and_state() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    state.record("consent", True, quote="yes")
    state.request_advance()
    prompt = build_system_prompt(state)
    assert "Sarah" in prompt                                  # persona included
    assert "Which state is your RN license in" in prompt      # current step's ask
    assert "consent" in prompt                                # filled field listed
    assert "pay_expectation" in prompt                        # remaining coverage listed
