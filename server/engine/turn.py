"""The streaming LLM turn: speech now, provisional extraction alongside.

## How this works
We speak the Responses API over raw SSE (httpx) - the project's raw-protocol
ethos; stream.py holds the typed-event assembly and prompt.py the prompts.
Two requests per caller turn:
- respond(): tool-free, so its first text streams straight to TTS. An async
  generator of SENTENCES - the Speaker starts TTS on sentence one while the
  model is still writing sentence three. llm_ttft is the first TEXT delta.
  History is persisted only under reply ownership and only after a
  speculative draft's commit gate opens, so a discarded draft leaves nothing.
- extract(): the concurrent evidence pass with tools, applied through
  evidence.apply_tools() under TURN ownership (a barge-in must not erase what
  the caller already said). Measured: it does not slow speech (+12 ms, n=12).
One httpx client per call; warm_connection() opens it during the TTS-only
greeting so the caller's first answer does not pay the ~660 ms handshake.
"""
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from server.config import Settings
from server.engine.evidence import apply_tools
from server.engine.plan import InterviewState
from server.engine.prompt import (
    TOOLS,
    build_extraction_instructions,
    build_instructions,
    cacheable_input,
    fallback_line,
)
from server.engine.stream import EngineTimeout, SentenceChunker, StreamAssembler, ToolCall

log = logging.getLogger(__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
# Cheapest authenticated GET on the same host: used only to open the
# connection so the first real request does not pay the handshake.
MODELS_URL = "https://api.openai.com/v1/models"
# Stage timeouts: connect bounds the handshake, read bounds the gap between
# streamed chunks — a stalled stream fails typed instead of hanging a turn.
ENGINE_CONNECT_TIMEOUT_SEC = 10.0
ENGINE_READ_TIMEOUT_SEC = 20.0


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

    async def warm_connection(self) -> None:
        """Establish the API connection before the first turn needs it.

        A fresh client pays DNS + TCP + TLS on its first request: measured
        ~660 ms (paired A/B, n=10, warmed faster 10/10). The client is
        per call, so that cost always landed on the caller's FIRST answer.
        The greeting is TTS-only and runs for seconds, which is the window to
        pay it invisibly. Best effort by design: on failure the call simply
        behaves as it did before.
        """
        try:
            await self._client.get(
                MODELS_URL,
                headers={"Authorization": f"Bearer {self._settings.openai_api_key}"})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.info("engine connection warm failed; first turn pays the handshake",
                     exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def _stream_lines(self, body: dict) -> AsyncIterator[str]:
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
        """Stream caller-facing speech from exactly one tool-free request."""
        current = is_current or (lambda: True)
        if not current():
            return
        state = self.state
        request_input = cacheable_input(state, user_text)
        body: dict[str, Any] = {
            "model": self._settings.turn_model,
            # Static prefix in instructions (cache-friendly); per-turn state
            # rides as a system input message ahead of the history window.
            "instructions": build_instructions(state.plan),
            "input": request_input,
            "stream": True,
            "store": False,
            "prompt_cache_key": "screener-" + hashlib.sha256(
                build_instructions(state.plan).encode("utf-8")
            ).hexdigest()[:24],
        }
        if self._settings.turn_model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "none"}  # TTFT is dominated by effort
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
        if not "".join(reply_parts).strip():
            line = fallback_line(state)
            reply_parts.append(line)
            yield line
        if commit_gate is not None:
            # Speculative speech/TTS may run warm, but history cannot persist
            # until the guessed caller turn really commits.
            await commit_gate.wait()
        if not current():
            return
        # Persisted only under reply ownership: a discarded speculative
        # generation leaves no phantom history entries.
        if (not state.history or state.history[-1].get("role") != "user"
                or state.history[-1].get("content") != user_text):
            state.add_history("user", user_text)
        state.add_history("assistant", "".join(reply_parts))
        details = (assembler.usage.get("input_tokens_details")
                   or assembler.usage.get("prompt_tokens_details") or {})
        self.last_cached_tokens = int(details.get("cached_tokens") or 0)
        self.last_cache_write_tokens = int(details.get("cache_write_tokens") or 0)

    async def extract(
        self,
        user_text: str,
        is_valid: Callable[[], bool] | None = None,
        turn_id: int | None = None,
        generation_id: int | None = None,
    ) -> list[dict]:
        """Extract provisional evidence without delaying caller-facing speech.

        Callers invoke this only for a committed caller turn. `is_valid` is
        turn ownership, deliberately independent of agent playback ownership:
        interrupting Sarah must not erase evidence the caller already supplied.
        """
        valid = is_valid or (lambda: True)
        if not valid():
            return []
        state = self.state
        instructions = build_extraction_instructions(state.plan)
        body: dict[str, Any] = {
            "model": self._settings.turn_model,
            "instructions": instructions,
            "input": cacheable_input(state, user_text),
            "tools": TOOLS,
            "stream": True,
            "store": False,
            "prompt_cache_key": "screener-extract-" + hashlib.sha256(
                instructions.encode("utf-8")
            ).hexdigest()[:24],
        }
        if self._settings.turn_model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "none"}
        assembler = StreamAssembler()
        async for line in self._stream_lines(body):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            assembler.feed(json.loads(payload))
        if not valid():
            return []
        results = self._apply_tools(
            assembler.tool_calls, valid, turn_id=turn_id,
            generation_id=generation_id, source_text=user_text)
        self.last_tool_results = results
        return results

    def _apply_tools(
        self,
        calls: list[ToolCall],
        is_current: Callable[[], bool] | None = None,
        turn_id: int | None = None,
        generation_id: int | None = None,
        source_text: str = "",
    ) -> list[dict]:
        # call_id defaults to "unassigned" (see __init__); a stale generation
        # must be able to bail out before any identity is computed.
        return apply_tools(
            self.state, calls, call_id=getattr(self, "_call_id", "unassigned"),
            is_current=is_current,
            turn_id=turn_id, generation_id=generation_id, source_text=source_text)
