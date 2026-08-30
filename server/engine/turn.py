"""One bounded, goal-driven engine turn: Responses stream -> tools -> speech.

## How this works
We speak the Responses API over raw SSE (httpx) — consistent with the project's
raw-protocol ethos, and the typed events are the whole lesson:
  response.output_text.delta          -> spoken text, fed to the SentenceChunker
  response.output_item.added          -> a function_call item opens (name, ids)
  response.function_call_arguments.delta/.done -> its JSON args accumulate/finish
StreamAssembler turns that event soup into text deltas + completed ToolCalls.

Tool calls are validated by the backend and their outputs are returned to the
model when it emitted no speech. That continuation is essential: a tool-only
Responses turn is an intermediate function-calling step, not a complete spoken
reply. These live captures are provisional interview memory only; post-call
caller-utterance verification is the sole authority for scores and knockouts.

LlmEngine.respond() is an async generator of SENTENCES: text deltas stream into
the chunker (abbreviation + decimal guards) and each complete sentence is
yielded immediately — the Speaker starts TTS on sentence one while the model is
still writing sentence three. llm_ttft (request start -> first text delta)
is recorded per turn. The system prompt is rendered fresh every turn from
InterviewState, so the model always sees current coverage; the engine, not the
model, remains the authority on scope, evidence and termination (see plan.py).
"""
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import Settings
from server.engine.intents import EndCallIntent, classify_end_call_intent
from server.engine.plan import InterviewState

log = logging.getLogger(__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
# Stage timeouts: connect bounds the handshake, read bounds the gap between
# streamed chunks — a stalled stream fails typed instead of hanging a turn.
ENGINE_CONNECT_TIMEOUT_SEC = 10.0
ENGINE_READ_TIMEOUT_SEC = 20.0
_ABBREVIATIONS = ("dr.", "st.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "vs.", "etc.")
_TERMINAL = ".!?"

TOOLS = [
    {
        "type": "function",
        "name": "record_answer",
        "description": "Record one answered field with the caller's verbatim words.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "value", "quote"],
            "properties": {
                "field": {"type": "string", "description": "Plan field name."},
                "value": {"type": "string",
                          "description": "The answer; lists comma-separated, booleans true/false."},
                "quote": {"type": "string",
                          "description": "Verbatim words the caller said."},
            },
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": (
            "Request a graceful end after the caller asks to stop or all "
            "interview objectives are covered. The backend validates and owns "
            "the actual session transition."),
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reason", "closing_message"],
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": [
                        "candidate_requested", "interview_complete",
                        "limit_reached",
                    ],
                },
                "closing_message": {
                    "type": "string",
                    "description": (
                        "One short final statement, with no question, to speak "
                        "before the backend closes the call."),
                },
            },
        },
    },
]


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""

    def push(self, delta: str):
        self._buf += delta
        while True:
            cut = self._find_boundary()
            if cut is None:
                return
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                yield sentence

    def _find_boundary(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if ch not in _TERMINAL:
                continue
            after = self._buf[i + 1: i + 2]
            if after and not after.isspace():
                continue  # mid-token: decimals like "4.5", or no space yet
            if not after:
                return None  # boundary may still be an abbreviation; wait for more
            head = self._buf[: i + 1]
            last_word = head.rsplit(None, 1)[-1].lower() if head.split() else ""
            if last_word in _ABBREVIATIONS:
                continue
            return i + 1
        return None

    def flush(self) -> str | None:
        tail = self._buf.strip()
        self._buf = ""
        return tail or None


class EngineStreamError(RuntimeError):
    """The Responses stream reported a failure event (arrives inside HTTP 200)."""


class EngineTimeout(EngineStreamError):
    """The provider exceeded a stage timeout (connect or inter-chunk read)."""


@dataclass
class ToolCall:
    name: str
    call_id: str
    arguments: dict | None  # None = args were malformed JSON


class StreamAssembler:
    def __init__(self) -> None:
        self.tool_calls: list[ToolCall] = []
        self._open: dict[str, dict] = {}  # item_id -> {name, call_id, buf}
        self.usage: dict = {}

    def feed(self, event: dict) -> list[str]:
        """Consume one typed event; return any text deltas it carried."""
        match event.get("type"):
            case "error":
                err = event.get("error", {})
                raise EngineStreamError(err.get("message", "stream error"))
            case "response.failed" | "response.incomplete":
                err = event.get("response", {}).get("error") or {}
                raise EngineStreamError(err.get("message", "response failed"))
            case "response.completed":
                self.usage = event.get("response", {}).get("usage") or {}
            case "response.output_text.delta":
                return [event.get("delta", "")]
            case "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    self._open[item["id"]] = {"name": item.get("name", ""),
                                              "call_id": item.get("call_id", ""),
                                              "buf": item.get("arguments", "")}
            case "response.function_call_arguments.delta":
                entry = self._open.get(event.get("item_id", ""))
                if entry is not None:
                    entry["buf"] += event.get("delta", "")
            case "response.function_call_arguments.done":
                entry = self._open.pop(event.get("item_id", ""), None)
                if entry is not None:
                    raw = event.get("arguments") or entry["buf"]
                    try:
                        args: dict | None = json.loads(raw)
                    except json.JSONDecodeError:
                        args = None
                    self.tool_calls.append(ToolCall(name=entry["name"],
                                                    call_id=entry["call_id"],
                                                    arguments=args))
        return []


def fallback_line(state: InterviewState) -> str:
    """Last-resort speech if both model passes produce no caller-facing text."""
    if state.end_call_request is not None:
        return state.end_call_request.closing_message
    if state.remaining:
        return "I'm sorry, I lost my place for a moment. What would you like me to clarify?"
    return "Thanks, that covers everything I needed for this screening."


def build_instructions(plan) -> str:
    """The STATIC per-call prefix (persona + rules). Kept byte-identical across
    turns so provider prompt caching can hit; everything that changes per turn
    lives in build_state_block and travels as an input message instead."""
    objectives = "\n".join(
        f"- {s.field} ({s.type}): {s.objective}" for s in plan.steps)
    prohibited = ", ".join(plan.boundaries.prohibited_topics) or "none configured"
    one_at_a_time = "yes" if plan.boundaries.ask_one_question_at_a_time else "no"
    return f"""{plan.persona}

You are conducting an open but bounded screening interview. You choose every
substantive question and its wording; there is no scripted question order.

Evidence objectives:
{objectives}

Boundaries:
- Maximum caller turns: {plan.boundaries.max_turns}
- Maximum duration: {plan.boundaries.max_duration_minutes:g} minutes
- Ask one question at a time: {one_at_a_time}
- Prohibited topics: {prohibited}

Rules:
- Respond to the caller's latest meaning first. Evidence objectives are coverage
  goals, not a script, required order, or whitelist of allowed conversation.
- Handle questions, corrections, uncertainty, requests for clarification, and
  brief role-relevant conversation before naturally returning to screening.
- If the caller asks what you need from them, explain the relevant screening
  goal plainly instead of using a generic prompt or immediately re-asking it.
- When it is natural to continue collecting evidence, choose the most useful
  question yourself. It may clarify context rather than directly fill a field.
- Do not repeat established information. If part of a compound topic was
  answered, ask only for what remains missing.
- Ask ONE concise question at a time. Keep speech to one or two short sentences.
- Avoid empty acknowledgements such as "Okay", "Got it", or "Alright". Only
  acknowledge something when the acknowledgement adds useful meaning.
- ALWAYS include a spoken reply for the caller. Never respond with only tool
  calls — a silent turn is a broken phone call.
- Every time the caller answers, call record_answer with the field name, the
  answer, and their VERBATIM words as the quote. If they volunteer other
  fields' answers, record those too.
- If the caller asks to stop, speak only a brief goodbye and call end_call with
  reason candidate_requested. Do not ask another interview question.
- When every objective is covered, briefly close and call end_call with reason
  interview_complete.
- Never invent answers. Never promise pay, benefits, or hiring decisions.
- A system message states what is established and what remains; it never gives
  you a required question or question order."""


def build_state_block(state: InterviewState) -> str:
    plan = state.plan
    # Quotes are truncated in the PROMPT only (full text stays in state for
    # post-call verification) — the prompt must not grow with caller verbosity.
    filled = "\n".join(f"- {name}: {rec.value!r} (they said: \"{rec.quote[:60]}\")"
                       for name, rec in state.fields.items()) or "- none yet"
    remaining = "\n".join(
        f"- {s.field} ({s.type}): {s.objective}"
        for s in state.remaining) or "- none"
    turns_left = max(0, plan.boundaries.max_turns - state.caller_turn_count)
    seconds_left = max(
        0.0, plan.boundaries.max_duration_minutes * 60 - state.elapsed_seconds)
    if state.end_call_request is not None:
        objective = "Termination is already requested. Do not ask another question."
    elif turns_left == 0 or seconds_left == 0:
        objective = (
            "The interview limit is reached. Ask no question. Give a brief "
            "closing statement and call end_call with reason limit_reached.")
    elif state.remaining:
        objective = (
            "Treat the missing evidence as coverage goals, not a questionnaire. "
            "Respond to the caller's latest intent first. When natural, continue "
            "with one useful question that you formulate yourself; it may answer "
            "or clarify the conversation before pursuing missing evidence.")
    else:
        objective = (
            "All evidence objectives are covered. Give a brief closing statement "
            "and call end_call with reason interview_complete.")
    return f"""Recorded so far:
{filled}
Evidence objectives still missing:
{remaining}

Runtime budget: {turns_left} caller turns and {seconds_left:.0f} seconds remain.

{objective}"""


def build_system_prompt(state: InterviewState) -> str:
    """Composition kept for tests and offline inspection; the live request
    sends the two halves separately for cacheability."""
    return build_instructions(state.plan) + "\n\n" + build_state_block(state)


class LlmEngine:
    def __init__(self, settings: Settings, state: InterviewState,
                 call_id: str = "unassigned") -> None:
        self._settings = settings
        self.state = state
        self._call_id = call_id
        self.last_ttft_ms: float | None = None
        self.last_cached_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_tool_results: list[dict] = []
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(
            30.0, connect=ENGINE_CONNECT_TIMEOUT_SEC,
            read=ENGINE_READ_TIMEOUT_SEC))

    async def close(self) -> None:
        await self._client.aclose()

    async def _stream_lines(self, body: dict):
        """SSE lines with provider timeouts mapped to a typed EngineTimeout."""
        try:
            async with self._client.stream(
                "POST", RESPONSES_URL, json=body,
                headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
            ) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeError(f"responses api {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise EngineTimeout(
                f"engine stage timeout ({exc.__class__.__name__})") from exc

    async def respond(
        self,
        user_text: str,
        is_current: Callable[[], bool] | None = None,
        turn_id: int | None = None,
        generation_id: int | None = None,
        commit_gate: asyncio.Event | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply sentences as they stream; apply tool calls as they land."""
        current = is_current or (lambda: True)
        if not current():
            return
        state = self.state
        request_input = ([{"role": "system", "content": build_state_block(state)}]
                         + state.recent_history(8)
                         + [{"role": "user", "content": user_text}])
        body: dict[str, Any] = {
            "model": self._settings.turn_model,
            # Static prefix in instructions (cache-friendly); per-turn state
            # rides as a system input message ahead of the history window.
            "instructions": build_instructions(state.plan),
            "input": request_input,
            "tools": TOOLS,
            "stream": True,
            "store": False,
            "prompt_cache_key": "screener-" + hashlib.sha256(
                build_instructions(state.plan).encode("utf-8")
            ).hexdigest()[:24],
        }
        if self._settings.turn_model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "none"}  # TTFT is dominated by effort
        # NOTE: verbosity "low" was trialed here per the model guide and
        # reverted after a live A/B: it suppressed record_answer diligence
        # (shift_availability never recorded across four asks; loop). Short
        # replies are already enforced by the prompt's one-question rule.
        chunker = SentenceChunker()
        assembler = StreamAssembler()
        reply_parts: list[str] = []
        t0 = time.monotonic()
        self.last_ttft_ms = None
        async for line in self._stream_lines(body):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            deltas = assembler.feed(event)
            if deltas and self.last_ttft_ms is None:
                # TTFT means first TEXT token, not response.created or another
                # administrative SSE event.
                self.last_ttft_ms = (time.monotonic() - t0) * 1000
            for delta in deltas:
                if not current():
                    return
                reply_parts.append(delta)
                for sentence in chunker.push(delta):
                    yield sentence
        if not current():
            return
        tail = chunker.flush()
        if tail:
            yield tail
        if commit_gate is not None:
            # Speculation is pure computation until the caller turn commits.
            # Cancellation while waiting discards every pending side effect.
            await commit_gate.wait()
        if not current():
            return
        # Apply validated side effects only after speculative ownership commits.
        self.last_tool_results = self._apply_tools(
            assembler.tool_calls, current, turn_id=turn_id,
            generation_id=generation_id, source_text=user_text)
        if not current():
            return
        if not "".join(reply_parts).strip() and assembler.tool_calls:
            if state.end_call_request is not None:
                # end_call already carries its short, model-authored closing.
                line = state.end_call_request.closing_message
                reply_parts.append(line)
                yield line
            else:
                # Responses function calling is request -> call -> output ->
                # response. A tool-only first pass expects this continuation;
                # canned speech here made the agent appear scripted.
                by_call_id = {
                    item.get("tool_call_id"): item
                    for item in self.last_tool_results
                }
                tool_exchange: list[dict[str, Any]] = []
                for call in assembler.tool_calls:
                    args = call.arguments if call.arguments is not None else {}
                    tool_exchange.append({
                        "type": "function_call", "call_id": call.call_id,
                        "name": call.name,
                        "arguments": json.dumps(args, separators=(",", ":")),
                    })
                for call in assembler.tool_calls:
                    result = by_call_id.get(call.call_id, {})
                    tool_exchange.append({
                        "type": "function_call_output", "call_id": call.call_id,
                        "output": json.dumps({
                            "applied": bool(result.get("applied")),
                            "reason": result.get("reason") or "recorded",
                        }, separators=(",", ":")),
                    })
                continuation = dict(body)
                # Keep the same tool definitions in context as prescribed by
                # the Responses flow, but force this bounded continuation to
                # produce caller-facing speech rather than start another loop.
                continuation["tool_choice"] = "none"
                continuation["input"] = request_input + tool_exchange + [{
                    "role": "system",
                    "content": (
                        build_state_block(state)
                        + "\nNow give the caller a natural spoken response to their "
                          "latest message. Do not mention tools or internal fields."
                    ),
                }]
                next_chunker = SentenceChunker()
                next_assembler = StreamAssembler()
                async for line in self._stream_lines(continuation):
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    deltas = next_assembler.feed(json.loads(payload))
                    if deltas and self.last_ttft_ms is None:
                        self.last_ttft_ms = (time.monotonic() - t0) * 1000
                    for delta in deltas:
                        if not current():
                            return
                        reply_parts.append(delta)
                        for sentence in next_chunker.push(delta):
                            yield sentence
                if not current():
                    return
                next_tail = next_chunker.flush()
                if next_tail:
                    yield next_tail
                if next_assembler.usage:
                    assembler.usage = next_assembler.usage
        if not "".join(reply_parts).strip():
            line = fallback_line(state)
            reply_parts.append(line)
            yield line
        if not current():
            return
        # Persisted only now, under the same ownership gate as the tools: a
        # discarded speculative generation leaves no phantom history entries.
        state.add_history("user", user_text)
        state.add_history("assistant", "".join(reply_parts))
        details = (assembler.usage.get("input_tokens_details")
                   or assembler.usage.get("prompt_tokens_details") or {})
        self.last_cached_tokens = int(details.get("cached_tokens") or 0)
        self.last_cache_write_tokens = int(details.get("cache_write_tokens") or 0)

    def _apply_tools(
        self,
        calls: list[ToolCall],
        is_current: Callable[[], bool] | None = None,
        turn_id: int | None = None,
        generation_id: int | None = None,
        source_text: str = "",
    ) -> list[dict]:
        """Validate, apply, and LEDGER every tool call. Failures are returned
        to orchestration as ledger entries (and warnings), never swallowed;
        duplicate tool_call_ids are idempotently skipped."""
        current = is_current or (lambda: True)
        ledger = self.state.tool_ledger
        seen = {e["idempotency_key"] for e in ledger
                if e.get("idempotency_key")}
        results: list[dict] = []
        # Function-call output order is model-controlled. Evidence mutations
        # must settle before an interview_complete request is validated, or an
        # end_call item emitted first can be rejected even though the same
        # response records the final objective. Stable sort preserves relative
        # order within each class and keeps the audit deterministic.
        ordered_calls = sorted(calls, key=lambda call: call.name == "end_call")
        for call in ordered_calls:
            if not current():
                return results
            tool_identity = call.call_id or hashlib.sha256(
                (call.name + json.dumps(call.arguments, sort_keys=True,
                                        default=str)).encode("utf-8")
            ).hexdigest()[:16]
            idempotency_key = f"{self._call_id}:{turn_id or 0}:{tool_identity}"
            execution_id = (f"{self._call_id}:{turn_id or 0}:"
                            f"{generation_id or 0}:{tool_identity}")
            entry = {"call_id": self._call_id,
                     "tool_call_id": call.call_id, "name": call.name,
                      "turn_id": turn_id, "generation_id": generation_id,
                     "execution_id": execution_id,
                     "idempotency_key": idempotency_key,
                      "arguments": call.arguments, "applied": False, "reason": ""}
            ledger.append(entry)
            results.append(entry)
            if idempotency_key in seen:
                entry["reason"] = "duplicate idempotency identity (skip)"
                continue
            seen.add(idempotency_key)
            if call.arguments is None:
                entry["reason"] = "malformed arguments"
                log.warning("malformed tool args for %s; skipped", call.name)
                continue
            if call.name == "record_answer":
                quote = str(call.arguments.get("quote") or "").strip()
                if not quote:
                    entry["reason"] = "empty quote: evidence required before mutation"
                    log.warning("record_answer without evidence rejected")
                    continue
                if not any(_quote_supported(quote, s) for s in
                           self._caller_sources(source_text)):
                    entry["reason"] = "quote not supported by any caller utterance"
                    log.warning("record_answer with unsupported evidence rejected")
                    continue
                entry["applied"] = self.state.record(
                    call.arguments.get("field", ""),
                    call.arguments.get("value"), quote)
                if not entry["applied"]:
                    entry["reason"] = "rejected by state validation"
                    log.warning("record_answer rejected by state validation")
            elif call.name == "end_call":
                reason = str(call.arguments.get("reason") or "")
                closing = str(call.arguments.get("closing_message") or "")
                if (reason == "candidate_requested"
                        and not any(classify_end_call_intent(source)
                                    == EndCallIntent.END
                                    for source in self._caller_sources(source_text))):
                    entry["reason"] = "candidate end request not supported by transcript"
                    continue
                if reason == "interview_complete" and self.state.remaining:
                    entry["reason"] = "interview objectives still missing"
                    continue
                if (reason == "limit_reached"
                        and self.state.caller_turn_count <
                        self.state.plan.boundaries.max_turns
                        and self.state.elapsed_seconds <
                        self.state.plan.boundaries.max_duration_minutes * 60):
                    entry["reason"] = "interview limit not reached"
                    continue
                entry["applied"] = self.state.request_end_call(reason, closing)
                if not entry["applied"]:
                    entry["reason"] = "invalid or duplicate end-call request"
            else:
                entry["reason"] = "unknown tool"
        return results


    def _caller_sources(self, source_text: str) -> list[str]:
        """Evidence may come from ANY committed caller utterance, not just the
        current one — the model legitimately records volunteered or delayed
        answers a turn late (live finding: single-utterance scoping caused
        systematic rejects and re-ask loops). History holds only committed,
        ownership-gated user turns, so this stays anchored to real speech."""
        sources = [h["content"] for h in self.state.history
                   if h.get("role") == "user"]
        sources.append(source_text)
        return [s for s in sources if s]


def _quote_supported(quote: str, source_text: str) -> bool:
    """Loose lexical anchoring: punctuation/case may differ, words may not."""
    import re

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    evidence = normalize(quote)
    source = normalize(source_text)
    return bool(evidence) and evidence in source
