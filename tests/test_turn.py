"""Sentence chunker, Responses-API event assembly, prompt rendering."""
import asyncio
import json
from pathlib import Path

from server.engine.plan import InterviewState, load_plan
import pytest

from server.engine.turn import (
    EngineStreamError,
    LlmEngine,
    SentenceChunker,
    StreamAssembler,
    ToolCall,
    build_system_prompt,
    fallback_line,
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


def test_system_prompt_targets_first_unfilled_even_if_cursor_lags() -> None:
    # Live regression: the model recorded answers without calling advance_step,
    # so the prompt kept re-asking consent. The objective must track need.
    state = InterviewState(load_plan(PLAN_PATH))
    state.record("consent", True, quote="yes")
    state.record("rn_license_state", "Texas", quote="Texas")
    prompt = build_system_prompt(state)                       # cursor still at consent
    assert "How many years of ICU experience" in prompt       # next ASKABLE step
    assert "rn_license_active" in prompt                      # still listed as needed


def test_fallback_line_speaks_the_plan_verbatim() -> None:
    # Live regression: tools-only responses produced silent turns. The plan's
    # own ask text is the deterministic never-silent fallback.
    state = InterviewState(load_plan(PLAN_PATH))
    state.record("consent", True, quote="yes")
    assert fallback_line(state) == "Which state is your RN license in, and is it currently active?"
    fill = {"bool": True, "float": 1.0, "list": ["x"], "str": "x"}
    for s in state.plan.steps:
        state.record(s.field, fill[s.type], quote="q")
    assert "thank" in fallback_line(state).lower()            # done: wrap-up line


def test_stale_generation_cannot_apply_tool_side_effects() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    engine = object.__new__(LlmEngine)
    engine.state = state
    calls = [
        ToolCall("record_answer", "c1", {
            "field": "consent", "value": True, "quote": "yes"}),
        ToolCall("advance_step", "c2", {}),
    ]

    engine._apply_tools(calls, is_current=lambda: False)

    assert state.fields == {}
    assert state.step_idx == 0


# --- Phase 3: tool execution ledger ---

def _engine_with_state():
    from pathlib import Path
    from server.config import Settings
    from server.engine.plan import InterviewState, load_plan
    from server.engine.turn import LlmEngine
    plan = load_plan(Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml")
    state = InterviewState(plan)
    return LlmEngine(Settings(_env_file=None, openai_api_key="k"), state), state


def _tc(call_id, name, args):
    from server.engine.turn import ToolCall
    return ToolCall(name=name, call_id=call_id, arguments=args)


def test_ledger_records_applied_and_rejected_calls() -> None:
    engine, state = _engine_with_state()
    engine._apply_tools([
        _tc("c1", "record_answer", {"field": "consent", "value": "true", "quote": "yes go ahead"}),
        _tc("c2", "record_answer", {"field": "not_a_field", "value": "x", "quote": "q"}),
        _tc("c3", "record_answer", {"field": "icu_years", "value": "5", "quote": ""}),
        _tc("c4", "advance_step", {}),
    ], turn_id=3, generation_id=7, source_text="yes go ahead q")
    ledger = state.tool_ledger
    assert [e["applied"] for e in ledger] == [True, False, False, True]
    assert ledger[1]["reason"] == "rejected by state validation"
    assert "quote" in ledger[2]["reason"]          # evidence required before mutation
    assert state.fields.keys() == {"consent"}      # empty-quote record never mutated
    assert all(e["turn_id"] == 3 and e["generation_id"] == 7 for e in ledger)


def test_ledger_skips_duplicate_tool_call_ids() -> None:
    engine, state = _engine_with_state()
    call = _tc("dup", "record_answer",
               {"field": "consent", "value": "true", "quote": "yes"})
    engine._apply_tools([call], turn_id=2, generation_id=4, source_text="yes")
    engine._apply_tools([call], turn_id=2, generation_id=5,
                        source_text="yes")  # replayed in a replacement generation
    applied = [e for e in state.tool_ledger if e["applied"]]
    skipped = [e for e in state.tool_ledger if "duplicate" in e["reason"]]
    assert len(applied) == 1 and len(skipped) == 1
    assert applied[0]["idempotency_key"] == skipped[0]["idempotency_key"]
    assert applied[0]["execution_id"] != skipped[0]["execution_id"]


def test_ledger_rejects_quote_not_in_committed_utterance() -> None:
    engine, state = _engine_with_state()
    results = engine._apply_tools([
        _tc("invented", "record_answer",
            {"field": "consent", "value": True, "quote": "absolutely yes"})
    ], turn_id=1, generation_id=2, source_text="No, I do not consent.")
    assert not results[0]["applied"]
    assert "caller utterance" in results[0]["reason"]
    assert state.fields == {}
    assert results[0]["call_id"] == "unassigned"
    assert results[0]["idempotency_key"].endswith(":1:invented")
    assert results[0]["execution_id"].endswith(":1:2:invented")


def test_speculative_engine_waits_for_commit_before_tools_and_history() -> None:
    engine, state = _engine_with_state()
    gate = asyncio.Event()

    async def stream(_body):
        events = _events_for_tool_call()
        for event in events:
            yield "data: " + json.dumps(event)
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        task = asyncio.create_task(_collect())
        await asyncio.sleep(0)
        assert state.fields == {}
        assert state.tool_ledger == []
        assert state.recent_history(8) == []
        gate.set()
        lines = await task
        assert lines
        assert state.fields["icu_years"].value == 5.0
        assert len(state.recent_history(8)) == 2
        await engine.close()

    async def _collect() -> list[str]:
        return [line async for line in engine.respond(
            "I have five years", turn_id=4, generation_id=9,
            commit_gate=gate)]

    asyncio.run(run())


def test_response_request_has_cache_key_low_verbosity_and_real_text_ttft() -> None:
    engine, _state = _engine_with_state()
    captured = {}

    async def stream(body):
        captured.update(body)
        yield 'data: {"type":"response.created"}'
        yield 'data: {"type":"response.output_text.delta","delta":"Okay. "}'
        yield ('data: {"type":"response.completed","response":{"usage":'
               '{"input_tokens_details":{"cached_tokens":42}}}}')
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        lines = [line async for line in engine.respond("yes")]
        assert lines == ["Okay."]
        assert engine.last_ttft_ms is not None
        assert engine.last_cached_tokens == 42
        assert captured["prompt_cache_key"].startswith("screener-")
        # verbosity "low" was reverted after a live A/B: it suppressed
        # record_answer diligence (fields went unrecorded; re-ask loop).
        assert "verbosity" not in captured.get("text", {})
        await engine.close()

    asyncio.run(run())


def test_evidence_from_an_earlier_committed_turn_is_accepted() -> None:
    # Live regression: the model records volunteered/delayed answers one turn
    # late, quoting an EARLIER caller utterance verbatim. Single-utterance
    # evidence scoping rejected those and caused re-ask loops.
    engine, state = _engine_with_state()
    state.add_history("user", "Honestly what I prefer is Nights are fine for me.")
    state.add_history("assistant", "Understood.")
    engine._apply_tools(
        [_tc("late1", "record_answer",
             {"field": "shift_availability", "value": "true",
              "quote": "Nights are fine for me."})],
        source_text="Somewhere around fifty five dollars an hour.")
    entry = state.tool_ledger[-1]
    assert entry["applied"] is True, entry["reason"]
