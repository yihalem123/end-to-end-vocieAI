"""ReplyController: everything that happens after a caller turn commits. Phase 4.

## How this works
Owns the reply side of a call so call.py stays a thin event router. A
GenerationSupervisor gives each turn one monotonic generation and serializes
replacement: invalidate, cancel, await cleanup, then start. Every audio send,
LLM mutation and transcript append checks ownership. Browser clear/drain acks
carry the same generation id; delayed acks are ignored. Voice completion waits
for playback_drained, so barge-in remains armed while buffered audio is audible.
- the ENGINE: LlmEngine (Responses API, plans/icu_nurse.yaml) when an OpenAI key
  exists, else the Phase 3 StubEngine. Both produce a stream of sentences behind
  one interface (_sentences_for).
- the SPEAKER + multi-context TTS (when an ElevenLabs key exists), pre-warmed at
  call setup; without it, replies are text-only.
- the BARGE-IN GUARD, which call.py's event loop ticks.
on_turn() starts one supervised reply worker: engine sentences stream through
the Speaker (TTS starts on sentence one while the LLM writes sentence three).
Voice errors fail that generation closed; replaying the text as a second reply
would violate at-most-once semantics. on_chat() uses the same serialized path.
Metrics per reply: llm_ttft from the engine, tts_ttfb / first_audio /
turn_latency from the Speaker.
"""
import asyncio
import logging
from contextlib import suppress
from pathlib import Path

from server.config import Settings
from server.engine.plan import InterviewState, load_plan
from server.engine.stub import StubEngine
from server.engine.turn import LlmEngine
from server.metrics import registry
from server.realtime.bargein import BargeInGuard
from server.realtime.endpoint import TurnComplete
from server.realtime.protocol import encode_audio_frame
from server.realtime.speaker import Speaker
from server.realtime.supervisor import GenerationSupervisor, GenerationToken
from server.realtime.tts import MultiContextTts

log = logging.getLogger(__name__)

# Perceived-latency filler, ON DEMAND: at turn commit we give the engine a
# patience window; if its first sentence arrives in time the caller hears only
# the real reply, and only a slow turn gets covered by an acknowledgment.
# Rotated for variety; skipped on the first exchange (an "Okay." before the
# greeting reads wrong).
FILLERS = ("Okay.", "Got it.", "Alright.", "Thanks.")
FILLER_PATIENCE_SEC = 0.6
SENTENCE_QUEUE_SIZE = 8  # bounded LLM -> TTS handoff; producer backpressures
DRAIN_ACK_MARGIN_SEC = 1.0
DRAIN_ACK_MAX_SEC = 5.0


def wants_filler(interview) -> bool:
    return interview is not None and bool(interview.fields)


def drain_wait_seconds(playback_remaining_sec: float) -> float:
    """Bound a remote drain acknowledgement by expected audio plus margin."""
    expected = max(0.0, playback_remaining_sec) + DRAIN_ACK_MARGIN_SEC
    return min(DRAIN_ACK_MAX_SEC, max(DRAIN_ACK_MARGIN_SEC, expected))


async def wait_for_playback_drain(
    drained: asyncio.Event,
    generation_id: int,
    timeout_sec: float,
) -> bool:
    try:
        await asyncio.wait_for(drained.wait(), timeout=timeout_sec)
        return True
    except TimeoutError:
        log.warning(
            "playback_drained ack timed out after %.2fs for generation %d; "
            "continuing fail-open",
            timeout_sec,
            generation_id,
        )
        return False


async def overlap_stream(filler: str, rest, patience_sec: float = FILLER_PATIENCE_SEC):
    """Run `rest` (the engine stream) in a pump task immediately; if its first
    item beats `patience_sec`, yield only real sentences — otherwise yield the
    filler to cover the gap, then the real reply as it lands. Errors from the
    pump re-raise in the consumer; cancelling the consumer (barge-in) cancels
    the pump and with it the engine request."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=SENTENCE_QUEUE_SIZE)

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
        with suppress(asyncio.CancelledError):
            await task


class ReplyController:
    def __init__(self, send_json, send_bytes, settings: Settings, state,
                 metric_prefix: str = "") -> None:
        self._send = send_json
        self._send_bytes = send_bytes
        self._metric_prefix = metric_prefix  # "flux_" tags the A/B in /metrics
        self._state = state  # realtime CallState: conversation log lives there
        self.guard = BargeInGuard()
        self._supervisor = GenerationSupervisor()
        self._drained: dict[int, asyncio.Event] = {}
        self._active_voice_generation: int | None = None
        self._tts: MultiContextTts | None = None
        self._prewarm: asyncio.Task | None = None
        self._speaker: Speaker | None = None
        if settings.elevenlabs_api_key:
            self._tts = MultiContextTts(settings.elevenlabs_api_key,
                                        settings.elevenlabs_voice_id)
            self._speaker = Speaker(self._send_audio, self._tts)
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

    async def _send_audio(self, generation_id: int, frame: bytes) -> None:
        token = self._supervisor.current
        if token is None or token.generation_id != generation_id:
            raise asyncio.CancelledError
        await self._send_bytes(encode_audio_frame(generation_id, frame))

    async def _sentences_for(self, token: GenerationToken, transcript: str):
        is_current = lambda: self._supervisor.is_current(token)
        if not is_current():
            return
        if isinstance(self._engine, LlmEngine):
            async for sentence in self._engine.respond(transcript, is_current=is_current):
                if not is_current():
                    return
                yield sentence
        else:
            for sentence in self._engine.reply(transcript):
                if not is_current():
                    return
                yield sentence

    # --- voice path ---

    async def on_turn(self, turn: TurnComplete, turn_id: int) -> None:
        await self._replace_current()
        if self._speaker is None:
            await self._supervisor.start(
                turn_id, lambda token: self._reply_text_only(token, turn.transcript))
            return
        token = await self._supervisor.start(
            turn_id, lambda owned: self._speak_reply(owned, turn))
        self._drained[token.generation_id] = asyncio.Event()
        self._active_voice_generation = token.generation_id
        self.guard.on_agent_audio_start()

    async def _speak_reply(self, token: GenerationToken, turn: TurnComplete) -> None:
        assert self._speaker is not None
        is_current = lambda: self._supervisor.is_current(token)
        sentences = self._sentences_for(token, turn.transcript)
        if wants_filler(self.interview):
            filler = FILLERS[self._filler_idx % len(FILLERS)]
            self._filler_idx += 1
            sentences = overlap_stream(filler, sentences)
        try:
            timings = await self._speaker.speak(
                sentences, turn.commit_t, turn.vad_stop_t,
                token.generation_id, is_current)
            if not is_current():
                return
            await self._send({"type": "audio_end", "turn_id": token.turn_id,
                              "generation_id": token.generation_id})
            drained = self._drained.setdefault(token.generation_id, asyncio.Event())
            remaining = float(timings.pop("_playback_remaining_sec", 0.0))
            await wait_for_playback_drain(
                drained,
                token.generation_id,
                drain_wait_seconds(remaining),
            )
            if not is_current():
                return
        except asyncio.CancelledError:
            raise  # barge-in/hangup: truncation and teardown own the rest
        except Exception:
            log.exception("voice reply failed; generation stopped")
            if is_current():
                self.guard.on_agent_audio_end()
                self._active_voice_generation = None
                await self._send({"type": "error", "message": "voice reply failed"})
            return
        finally:
            self._drained.pop(token.generation_id, None)
        if not is_current():
            return
        if isinstance(self._engine, LlmEngine) and self._engine.last_ttft_ms:
            timings["llm_ttft_ms"] = self._engine.last_ttft_ms
        registry.record_turn(**{self._metric_prefix + k: v
                                for k, v in timings.items() if k.endswith("_ms")})
        text = self._speaker.text(token.generation_id)
        self._state.conversation.append({
            "role": "agent", "text": text, "interrupted": False,
            "turn_id": token.turn_id, "generation_id": token.generation_id,
        })
        await self._send({"type": "agent", "text": text,
                          "interrupted": False, "audio": True,
                          "turn_id": token.turn_id,
                          "generation_id": token.generation_id})
        self.guard.on_agent_audio_end()
        self._active_voice_generation = None

    async def _reply_text_only(self, token: GenerationToken, transcript: str) -> None:
        is_current = lambda: self._supervisor.is_current(token)
        try:
            sentences = [s async for s in self._sentences_for(token, transcript)]
        except Exception:
            log.exception("engine failed")
            if is_current():
                await self._send({"type": "error", "message": "engine failed — see logs"})
            return
        if not is_current():
            return
        text = " ".join(sentences)
        if isinstance(self._engine, LlmEngine) and self._engine.last_ttft_ms:
            registry.record_turn(
                **{self._metric_prefix + "llm_ttft_ms": self._engine.last_ttft_ms})
        self._state.conversation.append({
            "role": "agent", "text": text, "interrupted": False,
            "turn_id": token.turn_id, "generation_id": token.generation_id,
        })
        await self._send({"type": "agent", "text": text, "interrupted": False,
                          "audio": False, "turn_id": token.turn_id,
                          "generation_id": token.generation_id})

    # --- text mode (chat box drives the engine without audio) ---

    async def on_chat(self, text: str, turn_id: int) -> None:
        await self._replace_current()
        await self._supervisor.start(
            turn_id, lambda token: self._reply_text_only(token, text))

    # --- barge-in ---

    async def _replace_current(self) -> None:
        if self._supervisor.current is None:
            return
        if self._active_voice_generation is not None:
            await self.interrupt_current(replacing=True)
        else:
            await self._supervisor.cancel_current()

    async def interrupt_current(self, replacing: bool = False) -> int | None:
        current = self._supervisor.current
        if current is None:
            return None
        if self._active_voice_generation != current.generation_id:
            await self._supervisor.cancel_current()
            return None
        token, task = self._supervisor.begin_interrupt()
        assert token is not None
        await self._send({"type": "clear", "turn_id": token.turn_id,
                          "generation_id": token.generation_id})
        await self._supervisor.wait_cancelled(task)
        if replacing:
            self.guard.on_agent_audio_end()
            self._active_voice_generation = None
        return token.generation_id

    async def on_cleared(self, generation_id: int, played_samples: int) -> None:
        if self._speaker is None:
            return
        token = self._supervisor.resolve_clear(generation_id)
        if token is None:  # delayed acknowledgement for an old generation
            return
        spoken, _ = self._speaker.truncate(generation_id, played_samples)
        if spoken:
            self._state.conversation.append({
                "role": "agent", "text": spoken, "interrupted": True,
                "turn_id": token.turn_id, "generation_id": token.generation_id,
            })
            await self._send({"type": "agent", "text": spoken, "interrupted": True,
                              "audio": True, "turn_id": token.turn_id,
                              "generation_id": token.generation_id})
        self.guard.on_agent_audio_end()
        self._active_voice_generation = None

    async def on_playback_drained(self, generation_id: int) -> None:
        current = self._supervisor.current
        if (current is None or current.generation_id != generation_id
                or self._active_voice_generation != generation_id):
            return
        event = self._drained.get(generation_id)
        if event is not None:
            event.set()

    async def on_playback_overflow(self, generation_id: int,
                                   played_samples: int) -> None:
        current = self._supervisor.current
        owns_current = current is not None and current.generation_id == generation_id
        if not owns_current and not self._supervisor.accepts_clear(generation_id):
            return
        if owns_current:
            token, task = self._supervisor.begin_interrupt()
            assert token is not None
            await self._supervisor.wait_cancelled(task)
        await self.on_cleared(generation_id, played_samples)
        await self._send({"type": "error", "message": "playback buffer overflow"})

    async def close(self) -> None:
        await self._supervisor.close()
        self.guard.on_agent_audio_end()
        if self._prewarm is not None and not self._prewarm.done():
            self._prewarm.cancel()
            with suppress(asyncio.CancelledError):
                await self._prewarm
        if self._tts is not None:
            await self._tts.close()
        if isinstance(self._engine, LlmEngine):
            await self._engine.close()
