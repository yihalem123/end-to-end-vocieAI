"""Sentence chunker, Responses-API event assembly, prompt rendering."""
import asyncio
import json
from pathlib import Path

from server.engine.plan import InterviewState, load_plan
import pytest

from server.engine.prompt import TOOLS, build_system_prompt, fallback_line
from server.engine.stream import EngineStreamError, SentenceChunker, StreamAssembler, ToolCall
from server.engine.turn import LlmEngine

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
    prompt = build_system_prompt(state)
    assert "Sarah" in prompt                                  # persona included
    assert "which US state issued" in prompt                  # objective, not script
    assert "there is no scripted question order" in prompt
    assert "consent" in prompt                                # filled field listed
    assert "pay_expectation" in prompt                        # remaining coverage listed


def test_system_prompt_exposes_all_unfilled_objectives_without_ordering() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    state.record("consent", True, quote="yes")
    state.record("rn_license_state", "Texas", quote="Texas")
    prompt = build_system_prompt(state)
    assert "rn_license_active" in prompt
    assert "icu_years" in prompt
    assert "coverage goals, not a questionnaire" in prompt
    assert "Respond to the caller's latest intent first" in prompt
    assert "Next question to get answered" not in prompt


def test_runtime_limit_changes_prompt_to_closing_only() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    state.caller_turn_count = state.plan.boundaries.max_turns
    prompt = build_system_prompt(state)
    assert "interview limit is reached" in prompt
    assert "Ask no question" in prompt


def test_limit_end_call_is_rejected_early_and_accepted_at_boundary() -> None:
    engine, state = _engine_with_state()
    call = _tc("limit", "end_call", {
        "reason": "limit_reached", "closing_message": "Thank you. Goodbye."})
    early = engine._apply_tools([call], source_text="hello")[-1]
    assert early["applied"] is False

    state.caller_turn_count = state.plan.boundaries.max_turns
    accepted = engine._apply_tools([
        _tc("limit2", "end_call", {
            "reason": "limit_reached", "closing_message": "Thank you. Goodbye."})
    ], source_text="hello")[-1]
    assert accepted["applied"] is True


def test_fallback_line_is_non_scripted_and_never_silent() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    state.record("consent", True, quote="yes")
    assert "clarify" in fallback_line(state).lower()
    assert "tell me a little more" not in fallback_line(state).lower()
    fill = {"bool": True, "float": 1.0, "list": ["x"], "str": "x"}
    for s in state.plan.steps:
        state.record(s.field, fill[s.type], quote="q")
    assert "thank" in fallback_line(state).lower()            # done: wrap-up line


def test_live_tool_surface_has_end_call_and_no_step_cursor() -> None:
    names = {tool["name"] for tool in TOOLS}
    assert names == {"record_answer", "end_call"}


def test_stale_generation_cannot_apply_tool_side_effects() -> None:
    state = InterviewState(load_plan(PLAN_PATH))
    engine = object.__new__(LlmEngine)
    engine.state = state
    calls = [
        ToolCall("record_answer", "c1", {
            "field": "consent", "value": True, "quote": "yes"}),
        ToolCall("end_call", "c2", {
            "reason": "candidate_requested", "closing_message": "Goodbye."}),
    ]

    engine._apply_tools(calls, is_current=lambda: False)

    assert state.fields == {}
    assert state.end_call_request is None


# --- tool execution ledger ---

def _engine_with_state():
    from pathlib import Path
    from server.config import Settings
    from server.engine.plan import InterviewState, load_plan
    from server.engine.turn import LlmEngine
    plan = load_plan(Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml")
    state = InterviewState(plan)
    return LlmEngine(Settings(_env_file=None, openai_api_key="k"), state), state


def _tc(call_id, name, args):
    from server.engine.stream import ToolCall
    return ToolCall(name=name, call_id=call_id, arguments=args)


def test_ledger_records_applied_and_rejected_calls() -> None:
    engine, state = _engine_with_state()
    engine._apply_tools([
        _tc("c1", "record_answer", {"field": "consent", "value": "true", "quote": "yes go ahead"}),
        _tc("c2", "record_answer", {"field": "not_a_field", "value": "x", "quote": "q"}),
        _tc("c3", "record_answer", {"field": "icu_years", "value": "5", "quote": ""}),
        _tc("c4", "end_call", {
            "reason": "candidate_requested",
            "closing_message": "Understood. Goodbye."}),
    ], turn_id=3, generation_id=7,
       source_text="yes go ahead q and I want to end this call")
    ledger = state.tool_ledger
    assert [e["applied"] for e in ledger] == [True, False, False, True]
    assert ledger[1]["reason"] == "rejected by state validation"
    assert "quote" in ledger[2]["reason"]          # evidence required before mutation
    assert state.fields.keys() == {"consent"}      # empty-quote record never mutated
    assert state.end_call_request.reason == "candidate_requested"
    assert all(e["turn_id"] == 3 and e["generation_id"] == 7 for e in ledger)


def test_end_call_requires_evidence_or_complete_coverage() -> None:
    engine, state = _engine_with_state()
    unsupported = engine._apply_tools([
        _tc("end1", "end_call", {
            "reason": "candidate_requested", "closing_message": "Goodbye."})
    ], source_text="I have five years.")[-1]
    assert unsupported["applied"] is False
    assert "not supported" in unsupported["reason"]

    premature = engine._apply_tools([
        _tc("end2", "end_call", {
            "reason": "interview_complete", "closing_message": "Thank you."})
    ], source_text="I have five years.")[-1]
    assert premature["applied"] is False
    assert "still missing" in premature["reason"]


def test_final_evidence_applies_before_completion_regardless_of_item_order() -> None:
    engine, state = _engine_with_state()
    fill = {"bool": True, "float": 1.0, "list": ["BLS"], "str": "known"}
    for step in state.plan.steps:
        if step.field != "pay_expectation":
            state.record(step.field, fill[step.type], quote="known")

    results = engine._apply_tools([
        _tc("finish-first", "end_call", {
            "reason": "interview_complete", "closing_message": "Thank you. Goodbye."}),
        _tc("final-answer", "record_answer", {
            "field": "pay_expectation", "value": "55 per hour", "quote": "55 per hour"}),
    ], source_text="55 per hour")

    assert [item["name"] for item in results] == ["record_answer", "end_call"]
    assert all(item["applied"] for item in results)
    assert state.end_call_request.reason == "interview_complete"


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


def test_speculative_engine_waits_for_commit_before_history() -> None:
    engine, state = _engine_with_state()
    gate = asyncio.Event()

    async def stream(_body):
        yield ('data: {"type":"response.output_text.delta",'
               '"delta":"Natural reply."}')
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        task = asyncio.create_task(_collect())
        await asyncio.sleep(0)
        assert state.recent_history(8) == []
        gate.set()
        lines = await task
        assert lines == ["Natural reply."]
        assert len(state.recent_history(8)) == 2
        await engine.close()

    async def _collect() -> list[str]:
        return [line async for line in engine.respond(
            "I have five years", turn_id=4, generation_id=9,
            commit_gate=gate)]

    asyncio.run(run())


def test_speech_is_one_tool_free_request_and_extraction_never_gates_it() -> None:
    engine, state = _engine_with_state()
    requests = []
    release_extraction = asyncio.Event()

    async def stream(body):
        requests.append(body)
        if "tools" in body:
            await release_extraction.wait()
            for event in _events_for_tool_call():
                yield "data: " + json.dumps(event)
        else:
            yield ('data: {"type":"response.output_text.delta","delta":'
                   '"Five years gives me useful context. What kinds of ICU patients "}')
            yield ('data: {"type":"response.output_text.delta","delta":'
                   '"have you worked with most recently?"}')
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        extraction = asyncio.create_task(engine.extract(
            "I have five years", turn_id=4, generation_id=9))
        lines = await asyncio.wait_for(_spoken(), timeout=0.2)
        assert lines == [
            "Five years gives me useful context.",
            "What kinds of ICU patients have you worked with most recently?",
        ]
        assert state.fields == {}, "background extraction is still deliberately blocked"
        release_extraction.set()
        results = await extraction
        assert state.fields["icu_years"].value == 5.0
        assert len(requests) == 2
        speech_request = next(body for body in requests if "tools" not in body)
        extraction_request = next(body for body in requests if "tools" in body)
        assert "tool_choice" not in speech_request
        assert extraction_request["tools"] == TOOLS
        assert results[0]["applied"] is True
        assert "tell me a little more" not in " ".join(lines).lower()
        await engine.close()

    async def _spoken() -> list[str]:
        return [line async for line in engine.respond(
            "I have five years", turn_id=4, generation_id=9)]

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
        assert "tools" not in captured
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


def test_chunker_does_not_split_inside_dotted_acronyms() -> None:
    # Live regression: free-form model text says "U.S." where the old scripted
    # plan text never did. "Which U.S. state issued your RN license?" was cut
    # into "Which U.S." + "state issued your RN license?" and synthesized as
    # two utterances — audibly chopped mid-sentence.
    ch = SentenceChunker()
    # Followed by more text, so the boundary resolves inside push() rather than
    # via the trailing-terminal hold - the split, if any, would show here.
    assert list(ch.push("Which U.S. state issued your RN license? Thanks.")) == [
        "Which U.S. state issued your RN license?"]
    assert ch.flush() == "Thanks."

    ch = SentenceChunker()
    out = list(ch.push("Call me at 9 a.m. tomorrow. Thanks."))
    assert out == ["Call me at 9 a.m. tomorrow."]
    assert ch.flush() == "Thanks."


def test_chunker_still_splits_normal_sentences_after_the_acronym_guard() -> None:
    ch = SentenceChunker()
    out = list(ch.push("Dr. Smith works at St. Mary's hospital. Great. "))
    assert out == ["Dr. Smith works at St. Mary's hospital.", "Great."]
    ch = SentenceChunker()
    assert list(ch.push("You mentioned 4.5 years. Noted. ")) == [
        "You mentioned 4.5 years.", "Noted."]


def test_volatile_state_rides_behind_the_stable_history_prefix() -> None:
    # Cache-friendly layout: anything that changes per turn must come AFTER
    # everything that does not. (This does not achieve cache hits on its own -
    # see _cacheable_input for the measured reasons - but the reverse ordering
    # makes them impossible, and state reads better next to the turn it
    # describes.)
    engine, state = _engine_with_state()
    captured = {}

    async def stream(body):
        captured.update(body)
        yield 'data: {"type":"response.output_text.delta","delta":"Okay."}'
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        state.add_history("user", "Yes, I consent.")
        state.add_history("assistant", "Thanks. Which state issued your license?")
        [line async for line in engine.respond("Delaware.")]
        roles = [item["role"] for item in captured["input"]]
        contents = [item["content"] for item in captured["input"]]

        # history first, then the volatile state block, then the newest turn
        assert roles[0] == "user" and contents[0] == "Yes, I consent."
        state_at = next(i for i, c in enumerate(contents) if "consent" in c
                        and roles[i] == "system")
        assert state_at == len(contents) - 2
        assert contents[-1] == "Delaware."
        await engine.close()

    asyncio.run(run())


def test_warm_connection_opens_the_api_connection_before_the_first_turn() -> None:
    # Measured (paired, n=10): a brand-new client pays ~660 ms of handshake on
    # its first Responses request, and the client is per call — so the caller's
    # FIRST answer always paid it. The greeting is TTS-only and runs ~14 s,
    # which is exactly the window to get the connection established.
    engine, _state = _engine_with_state()
    got: list[str] = []

    class FakeClient:
        async def get(self, url, headers=None):
            got.append(url)
            return None

    engine._client = FakeClient()

    async def run() -> None:
        await engine.warm_connection()
        assert got and got[0].startswith("https://api.openai.com/")

    asyncio.run(run())


def test_warm_connection_never_fails_a_call() -> None:
    engine, _state = _engine_with_state()

    class BrokenClient:
        async def get(self, url, headers=None):
            raise OSError("no route to host")

    engine._client = BrokenClient()

    async def run() -> None:
        await engine.warm_connection()      # best effort: must not raise

    asyncio.run(run())



def test_speech_request_can_opt_into_a_priority_tier_and_extraction_never_does() -> None:
    # The tier is billed at a premium and measured as a wash on the real prompt
    # (see config.py), so it is opt-in; when set, only the request the caller
    # is waiting on carries it.
    engine, state = _engine_with_state()
    engine._settings.openai_speech_service_tier = "priority"
    captured: list[dict] = []

    async def stream(body):
        captured.append(body)
        yield 'data: {"type":"response.output_text.delta","delta":"Okay."}'
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        [line async for line in engine.respond("yes")]
        await engine.extract("yes")
        speech, evidence = captured
        assert speech["service_tier"] == "priority"
        assert "service_tier" not in evidence
        await engine.close()

    asyncio.run(run())


def test_no_service_tier_is_sent_by_default() -> None:
    from server.config import Settings
    from server.engine.plan import InterviewState, load_plan
    from server.engine.turn import LlmEngine

    settings = Settings(_env_file=None, openai_api_key="k")
    assert settings.openai_speech_service_tier == "default"
    engine = LlmEngine(settings, InterviewState(load_plan(PLAN_PATH)))
    captured: list[dict] = []

    async def stream(body):
        captured.append(body)
        yield "data: [DONE]"

    engine._stream_lines = stream

    async def run() -> None:
        [line async for line in engine.respond("yes")]
        assert "service_tier" not in captured[0]
        await engine.close()

    asyncio.run(run())


def test_engine_client_speaks_http2(monkeypatch) -> None:
    import httpx
    from server.engine import turn as turn_module

    seen: dict = {}
    real = httpx.AsyncClient

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(turn_module.httpx, "AsyncClient", spy)
    engine, _state = _engine_with_state()
    assert seen.get("http2") is True
    asyncio.run(engine.close())


def test_extraction_instructions_accept_hedged_answers() -> None:
    # Live: "I guess so." was treated as uncertainty, never recorded, and the
    # night-shift question was asked three times in a row.
    from server.engine.plan import load_plan
    from server.engine.prompt import build_extraction_instructions

    text = build_extraction_instructions(load_plan(PLAN_PATH))
    assert "hedged but substantive answer is evidence" in text
    assert "true or false" in text


def test_speech_instructions_tell_the_model_a_hedge_is_an_answer() -> None:
    from server.engine.plan import load_plan
    from server.engine.prompt import build_instructions

    assert "hedged answer" in build_instructions(load_plan(PLAN_PATH))
