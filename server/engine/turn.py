"""One engine turn: OpenAI Responses API streaming -> tools applied -> sentences.

## How this works
We speak the Responses API over raw SSE (httpx) — consistent with the project's
raw-protocol ethos, and the typed events are the whole lesson:
  response.output_text.delta          -> spoken text, fed to the SentenceChunker
  response.output_item.added          -> a function_call item opens (name, ids)
  response.function_call_arguments.delta/.done -> its JSON args accumulate/finish
StreamAssembler turns that event soup into text deltas + completed ToolCalls.

Tools are ONE-WAY state mutations (record_answer, advance_step): we never send
tool outputs back, so one request per turn — no second round trip, which is the
latency trick that makes tool use viable in a voice loop. It also makes multiple
calls per response cancellation-safe (a barge-in that kills the turn mid-stream
leaves recorded fields recorded and the advance validated or not — state stays
consistent), so parallel tool calls stay enabled.

LlmEngine.respond() is an async generator of SENTENCES: text deltas stream into
the chunker (abbreviation + decimal guards) and each complete sentence is
yielded immediately — the Speaker starts TTS on sentence one while the model is
still writing sentence three. llm_ttft (request start -> first streamed event)
is recorded per turn. The system prompt is rendered fresh every turn from
InterviewState, so the model always sees current coverage; the engine, not the
model, remains the authority on step order (see plan.py).
"""
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import Settings
from server.engine.plan import InterviewState

log = logging.getLogger(__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
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
        "name": "advance_step",
        "description": "Signal the current question is fully answered.",
        "strict": True,
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": [], "properties": {}},
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


@dataclass
class ToolCall:
    name: str
    call_id: str
    arguments: dict | None  # None = args were malformed JSON


class StreamAssembler:
    def __init__(self) -> None:
        self.tool_calls: list[ToolCall] = []
        self._open: dict[str, dict] = {}  # item_id -> {name, call_id, buf}

    def feed(self, event: dict) -> list[str]:
        """Consume one typed event; return any text deltas it carried."""
        match event.get("type"):
            case "error":
                err = event.get("error", {})
                raise EngineStreamError(err.get("message", "stream error"))
            case "response.failed" | "response.incomplete":
                err = event.get("response", {}).get("error") or {}
                raise EngineStreamError(err.get("message", "response failed"))
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
    """Deterministic reply when the model returns tools but no text (a silent
    turn is never acceptable in voice). The plan's own words are the script."""
    if state.knocked_out:
        return ("Thanks for your time today. Unfortunately this role requires "
                "that, so we won't move forward — but thank you for talking with me.")
    step = state.next_askable
    if step is not None:
        return step.ask if step.ask else state.plan.consent
    needed = state.next_needed
    if needed is not None:  # only ask-less fields remain: confirm explicitly
        return f"One more thing to confirm: {needed.field.replace('_', ' ')}?"
    return "That's everything I needed — thank you for your time today!"


def build_system_prompt(state: InterviewState) -> str:
    plan = state.plan
    # Quotes are truncated in the PROMPT only (full text stays in state for
    # post-call verification) — the prompt must not grow with caller verbosity.
    filled = "\n".join(f"- {name}: {rec.value!r} (they said: \"{rec.quote[:60]}\")"
                       for name, rec in state.fields.items()) or "- none yet"
    remaining = ", ".join(s.field for s in plan.steps
                          if s.field not in state.fields) or "none"
    # Target NEED (first unfilled, askable step), not the cursor: the model may
    # record answers without signaling advance_step, and must not re-ask
    # stale steps. Ask-less steps are covered opportunistically or confirmed.
    step = state.next_askable
    if state.knocked_out:
        objective = ("The caller did not pass a required check "
                     f"({state.knocked_out}). Politely wrap up the call now.")
    elif step is not None:
        ask = step.ask if step.ask else plan.consent
        objective = f'Next question to get answered: "{ask}"'
    elif state.next_needed is not None:
        objective = (f"Only {state.next_needed.field} still needs confirming — "
                     "ask for it directly, then wrap up.")
    else:
        objective = "All questions are covered. Thank them and wrap up the call."
    return f"""{plan.persona}

You are conducting a structured screening interview. Rules:
- Ask ONE question at a time. Keep replies to one or two short sentences.
- ALWAYS include a spoken reply for the caller. Never respond with only tool
  calls — a silent turn is a broken phone call.
- Every time the caller answers, call record_answer with the field name, the
  answer, and their VERBATIM words as the quote. If they volunteer other
  fields' answers, record those too.
- When the current question is fully answered, call advance_step.
- Never invent answers. Never promise pay, benefits, or hiring decisions.

Recorded so far:
{filled}
Fields still needed: {remaining}

{objective}"""


class LlmEngine:
    def __init__(self, settings: Settings, state: InterviewState) -> None:
        self._settings = settings
        self.state = state
        self.last_ttft_ms: float | None = None
        self._client = httpx.AsyncClient(timeout=30)

    async def close(self) -> None:
        await self._client.aclose()

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """Yield reply sentences as they stream; apply tool calls as they land."""
        state = self.state
        state.add_history("user", user_text)
        body: dict[str, Any] = {
            "model": self._settings.turn_model,
            "instructions": build_system_prompt(state),
            "input": state.recent_history(8),
            "tools": TOOLS,
            "stream": True,
            "store": False,
        }
        if self._settings.turn_model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "none"}  # TTFT is dominated by effort
        chunker = SentenceChunker()
        assembler = StreamAssembler()
        reply_parts: list[str] = []
        t0 = time.monotonic()
        self.last_ttft_ms = None
        async with self._client.stream(
            "POST", RESPONSES_URL, json=body,
            headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
        ) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"responses api {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if self.last_ttft_ms is None:
                    self.last_ttft_ms = (time.monotonic() - t0) * 1000
                for delta in assembler.feed(event):
                    reply_parts.append(delta)
                    for sentence in chunker.push(delta):
                        yield sentence
        tail = chunker.flush()
        if tail:
            yield tail
        # Tools first so the fallback sees the state they just updated.
        self._apply_tools(assembler.tool_calls)
        if not "".join(reply_parts).strip():
            line = fallback_line(state)  # tools-only turn: never go silent
            reply_parts.append(line)
            yield line
        state.add_history("assistant", "".join(reply_parts))

    def _apply_tools(self, calls: list[ToolCall]) -> None:
        for call in calls:
            if call.arguments is None:
                log.warning("malformed tool args for %s; skipped", call.name)
                continue
            if call.name == "record_answer":
                ok = self.state.record(call.arguments.get("field", ""),
                                       call.arguments.get("value"),
                                       call.arguments.get("quote", ""))
                if not ok:
                    log.warning("record_answer rejected: %s", call.arguments)
            elif call.name == "advance_step":
                self.state.request_advance()
