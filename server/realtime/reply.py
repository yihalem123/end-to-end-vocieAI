"""ReplyController: everything that happens after a caller turn commits. Phase 4.

## How this works
Owns the reply side of a call so call.py stays a thin event router:
- the ENGINE: LlmEngine (Responses API, plans/icu_nurse.yaml) when an OpenAI key
  exists, else the Phase 3 StubEngine. Both produce a stream of sentences behind
  one interface (_sentences_for).
- the SPEAKER + multi-context TTS (when an ElevenLabs key exists), pre-warmed at
  call setup; without it, replies are text-only.
- the BARGE-IN GUARD, which call.py's event loop ticks.
on_turn() spawns a cancellable speak task per reply: engine sentences stream
through the Speaker (TTS starts on sentence one while the LLM writes sentence
three). Engine failure mid-reply falls back to text (scope guard: vendor
trouble degrades, never kills the call). on_chat() is the text-mode path — same
engine, no TTS — which is how the engine was developed before it had a voice.
Metrics per reply: llm_ttft from the engine, tts_ttfb / first_audio /
turn_latency from the Speaker.
"""
import asyncio
import logging
from pathlib import Path

from server.config import Settings
from server.engine.plan import InterviewState, load_plan
from server.engine.stub import StubEngine
from server.engine.turn import LlmEngine
from server.metrics import registry
from server.realtime.bargein import BargeInGuard
from server.realtime.endpoint import TurnComplete
from server.realtime.speaker import Speaker
from server.realtime.tts import MultiContextTts

log = logging.getLogger(__name__)

# Perceived-latency filler, ON DEMAND: at turn commit we give the engine a
# patience window; if its first sentence arrives in time the caller hears only
# the real reply, and only a slow turn gets covered by an acknowledgment.
# Rotated for variety; skipped on the first exchange (an "Okay." before the
# greeting reads wrong).
FILLERS = ("Okay.", "Got it.", "Alright.", "Thanks.")
FILLER_PATIENCE_SEC = 0.6


def wants_filler(interview) -> bool:
    return interview is not None and bool(interview.fields)


async def overlap_stream(filler: str, rest, patience_sec: float = FILLER_PATIENCE_SEC):
    """Run `rest` (the engine stream) in a pump task immediately; if its first
    item beats `patience_sec`, yield only real sentences — otherwise yield the
    filler to cover the gap, then the real reply as it lands. Errors from the
    pump re-raise in the consumer; cancelling the consumer (barge-in) cancels
    the pump and with it the engine request."""
    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in rest:
                await queue.put(item)
            await queue.put(None)
        except Exception as exc:  # noqa: BLE001 — carried to the consumer
            await queue.put(exc)

    task = asyncio.create_task(pump())
    try:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=patience_sec)
        except TimeoutError:
            yield filler  # engine is slow this turn: cover the silence
            item = await queue.get()
        while True:
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
            item = await queue.get()
    finally:
        task.cancel()


class ReplyController:
    def __init__(self, send_json, send_bytes, settings: Settings, state) -> None:
        self._send = send_json
        self._state = state  # realtime CallState: replies list lives there
        self.guard = BargeInGuard()
        self._speak_task: asyncio.Task | None = None
        self._tts: MultiContextTts | None = None
        self._prewarm: asyncio.Task | None = None
        self._speaker: Speaker | None = None
        if settings.elevenlabs_api_key:
            self._tts = MultiContextTts(settings.elevenlabs_api_key,
                                        settings.elevenlabs_voice_id)
            self._speaker = Speaker(send_bytes, self._tts)
            self._prewarm = asyncio.create_task(self._tts.ensure_connected())
        self._filler_idx = 0
        self._engine: LlmEngine | StubEngine
        if settings.openai_api_key:
            plan = load_plan(Path(settings.plan_path))
            self.interview = InterviewState(plan)
            self._engine = LlmEngine(settings, self.interview)
        else:
            self.interview = None
            self._engine = StubEngine()

    async def _sentences_for(self, transcript: str):
        if isinstance(self._engine, LlmEngine):
            async for sentence in self._engine.respond(transcript):
                yield sentence
        else:
            for sentence in self._engine.reply(transcript):
                yield sentence

    # --- voice path ---

    async def on_turn(self, turn: TurnComplete) -> None:
        if self._speaker is None:
            await self._reply_text_only(turn.transcript)
            return
        self.guard.on_agent_audio_start()
        self._speak_task = asyncio.create_task(self._speak_reply(turn))

    async def _speak_reply(self, turn: TurnComplete) -> None:
        assert self._speaker is not None
        sentences = self._sentences_for(turn.transcript)
        if wants_filler(self.interview):
            filler = FILLERS[self._filler_idx % len(FILLERS)]
            self._filler_idx += 1
            sentences = overlap_stream(filler, sentences)
        try:
            timings = await self._speaker.speak(
                sentences, turn.commit_t, turn.vad_stop_t)
        except asyncio.CancelledError:
            raise  # barge-in/hangup: truncation and teardown own the rest
        except Exception:
            log.exception("reply failed; degrading to text-only")
            self.guard.on_agent_audio_end()
            await self._reply_text_only(turn.transcript, already_tried_voice=True)
            return
        if isinstance(self._engine, LlmEngine) and self._engine.last_ttft_ms:
            timings["llm_ttft_ms"] = self._engine.last_ttft_ms
        registry.record_turn(**{k: v for k, v in timings.items() if k.endswith("_ms")})
        self.guard.on_agent_audio_end()
        text = " ".join(self._speaker.sentences)
        self._state.conversation.append({"role": "agent", "text": text, "interrupted": False})
        await self._send({"type": "agent", "text": text,
                          "interrupted": False, "audio": True})

    async def _reply_text_only(self, transcript: str,
                               already_tried_voice: bool = False) -> None:
        try:
            sentences = [s async for s in self._sentences_for(transcript)]
        except Exception:
            log.exception("engine failed")
            await self._send({"type": "error", "message": "engine failed — see logs"})
            return
        text = " ".join(sentences)
        if isinstance(self._engine, LlmEngine) and self._engine.last_ttft_ms:
            registry.record_turn(llm_ttft_ms=self._engine.last_ttft_ms)
        self._state.conversation.append({"role": "agent", "text": text, "interrupted": False})
        await self._send({"type": "agent", "text": text, "interrupted": False,
                          "audio": False})

    # --- text mode (chat box drives the engine without audio) ---

    async def on_chat(self, text: str) -> None:
        await self._reply_text_only(text)

    # --- barge-in ---

    def cancel_current(self) -> None:
        if self._speak_task is not None and not self._speak_task.done():
            self._speak_task.cancel()

    async def on_cleared(self, played_samples: int) -> None:
        if self._speaker is None:
            return
        spoken, _ = self._speaker.truncate(played_samples)
        self._state.conversation.append({"role": "agent", "text": spoken, "interrupted": True})
        await self._send({"type": "agent", "text": spoken, "interrupted": True,
                          "audio": True})

    async def close(self) -> None:
        self.cancel_current()
        if self._prewarm is not None and not self._prewarm.done():
            self._prewarm.cancel()
        if self._tts is not None:
            await self._tts.close()
        if isinstance(self._engine, LlmEngine):
            await self._engine.close()
