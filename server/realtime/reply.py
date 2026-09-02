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

from server.config import Settings
from server.engine.plan import InterviewPlan, InterviewState, load_plan_cached
from server.engine.stub import StubEngine
from server.engine.turn import LlmEngine, SentenceChunker
from server.metrics import registry
from server.realtime.bargein import BargeInGuard
from server.realtime.endpoint import TurnComplete
from server.realtime.protocol import encode_audio_frame
from server.realtime.speaker import Speaker
from server.realtime.supervisor import GenerationSupervisor, GenerationToken
from server.realtime.tts import MultiContextTts
from server.realtime.tts_aura import AuraTts

log = logging.getLogger(__name__)


def _spoken_eq(a: str, b: str) -> bool:
    """Allow only casing/space and final punctuation changes on promotion.

    Internal punctuation can change meaning ("No, nights" vs "No nights"), so
    it remains part of the identity check.
    """
    import re

    def norm(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".!?")

    return norm(a) == norm(b)

DRAIN_ACK_MARGIN_SEC = 1.0
DRAIN_ACK_MAX_SEC = 5.0


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


class ReplyController:
    def __init__(self, send_json, send_bytes, settings: Settings, state,
                 metric_prefix: str = "", call_id: str = "unassigned",
                 plan: InterviewPlan | None = None) -> None:
        self._send = send_json
        self._send_bytes = send_bytes
        self._metric_prefix = metric_prefix  # "flux_" tags the A/B in /metrics
        self._call_id = call_id
        self._state = state  # realtime CallState: conversation log lives there
        self.guard = BargeInGuard()
        self._supervisor = GenerationSupervisor()
        self._drained: dict[int, asyncio.Event] = {}
        self._spec: dict | None = None  # in-flight speculative generation
        self._active_voice_generation: int | None = None
        self._audible_generation: int | None = None
        self._browser_turn_anchors: dict[int, float] = {}
        # Evidence extraction is owned by committed caller turns, not agent
        # playback. Requests apply in turn order and never gate speech.
        self._committed_turns: set[int] = set()
        self._extraction_tasks: set[asyncio.Task] = set()
        self._extraction_tail: asyncio.Task | None = None
        self._closed = False
        self._tts: MultiContextTts | AuraTts | None = None
        self._prewarm: asyncio.Task | None = None
        self._speaker: Speaker | None = None
        # Providers are interchangeable behind Speaker's synthesize() surface;
        # TTS_PROVIDER picks one (aura rides the Deepgram key).
        if settings.tts_provider == "aura" and settings.deepgram_api_key:
            self._tts = AuraTts(settings.deepgram_api_key, settings.aura_model)
        elif settings.elevenlabs_api_key:
            self._tts = MultiContextTts(settings.elevenlabs_api_key,
                                        settings.elevenlabs_voice_id)
        if self._tts is not None:
            self._speaker = Speaker(self._send_audio, self._tts)
            self._prewarm = asyncio.create_task(self._tts.ensure_connected())
        plan = plan if plan is not None else load_plan_cached(str(settings.plan_path))
        self.interview = InterviewState(plan)
        self._engine: LlmEngine | StubEngine
        self._engine_warm: asyncio.Task | None = None
        if settings.openai_api_key:
            self._engine = LlmEngine(settings, self.interview, call_id=call_id)
            # The greeting is TTS-only, so the API connection is idle for
            # seconds: open it now and the caller's first answer stops paying
            # the ~660 ms handshake (measured paired, n=10).
            self._engine_warm = asyncio.create_task(self._engine.warm_connection())
        else:
            self._engine = StubEngine()

    @property
    def is_idle(self) -> bool:
        return self._supervisor.current is None

    async def _send_audio(self, generation_id: int, frame: bytes) -> None:
        token = self._supervisor.current
        if token is None or token.generation_id != generation_id:
            raise asyncio.CancelledError
        if self._audible_generation != generation_id:
            # Arm barge-in HERE, not when the generation starts: the caller can
            # only interrupt speech they can hear, and first audio trails the
            # generation by ~2 s (LLM + TTS connect). Arming early made room
            # noise cancel replies that were never heard — the call went silent
            # and looked dead. Caller speech before this point is not an
            # interruption; it commits a turn and replaces the reply normally.
            self._audible_generation = generation_id
            self.guard.on_agent_audio_start()
        await self._send_bytes(encode_audio_frame(generation_id, frame))

    async def _sentences_for(self, token: GenerationToken, transcript: str,
                             commit_gate: asyncio.Event | None = None):
        def is_current() -> bool:
            return self._supervisor.is_current(token)
        if not is_current():
            return
        if isinstance(self._engine, LlmEngine):
            async for sentence in self._engine.respond(
                    transcript, is_current=is_current,
                    turn_id=token.turn_id, generation_id=token.generation_id,
                    commit_gate=commit_gate):
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
        spec = self._spec
        if (spec is not None
                and self._supervisor.is_current(spec["token"])
                and spec["token"].turn_id == turn_id
                and _spoken_eq(spec["transcript"], turn.transcript)):
            # PROMOTE: the guessed turn is the committed turn. The generation
            # has been running since vad_stop; fill the timing anchors and
            # release the audio gate — everything downstream is already warm.
            self._spec = None
            token = spec["token"]
            log.info("speculation PROMOTED gen=%d", token.generation_id)
            spec["anchors"]["commit_t"] = turn.commit_t
            spec["anchors"]["vad_stop_t"] = turn.vad_stop_t
            self._drained[token.generation_id] = asyncio.Event()
            self._active_voice_generation = token.generation_id
            self._remember_browser_anchor(token.generation_id, turn.vad_stop_t)
            self._queue_extraction(token, turn.transcript)
            spec["release"].set()
            return
        await self._replace_current()
        if self._speaker is None:
            token = await self._supervisor.start(
                turn_id, lambda token: self._reply_text_only(token, turn.transcript))
            self._queue_extraction(token, turn.transcript)
            return
        token = await self._supervisor.start(
            turn_id, lambda owned: self._speak_reply(owned, turn))
        self._drained[token.generation_id] = asyncio.Event()
        self._active_voice_generation = token.generation_id
        self._remember_browser_anchor(token.generation_id, turn.vad_stop_t)
        self._queue_extraction(token, turn.transcript)

    async def _speak_reply(self, token: GenerationToken, turn: TurnComplete) -> None:
        sentences = self._voiced_sentences(token, turn.transcript)
        await self._deliver_voice(
            token, sentences,
            {"commit_t": turn.commit_t, "vad_stop_t": turn.vad_stop_t},
            record_metrics=True)

    def _voiced_sentences(self, token: GenerationToken, transcript: str,
                          commit_gate: asyncio.Event | None = None):
        # The model owns every spoken word. Server-injected acknowledgements
        # sounded repetitive and could contradict the response that followed.
        return self._sentences_for(token, transcript, commit_gate=commit_gate)

    async def _deliver_voice(self, token: GenerationToken, sentences,
                             anchors: dict, record_metrics: bool,
                             release: asyncio.Event | None = None) -> None:
        assert self._speaker is not None
        def is_current() -> bool:
            return self._supervisor.is_current(token)

        spoken: list[str] = []

        async def announce(sentence: str) -> None:
            """Grow the console transcript as the caller hears each sentence."""
            spoken.append(sentence)
            if is_current():
                await self._send({
                    "type": "agent_partial", "text": " ".join(spoken),
                    "turn_id": token.turn_id,
                    "generation_id": token.generation_id,
                })

        try:
            timings = await self._speaker.speak(
                sentences, anchors, token.generation_id, is_current, release,
                on_sentence=announce)
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
            if release is not None and not release.is_set():
                # A draft is not a caller-visible turn. Promotion will see that
                # ownership cleared and run one normal generation instead.
                log.info("speculative reply failed before promotion", exc_info=True)
                return
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
        if (record_metrics and isinstance(self._engine, LlmEngine)
                and self._engine.last_ttft_ms):
            timings["llm_ttft_ms"] = self._engine.last_ttft_ms
            timings["llm_cached_tokens"] = float(self._engine.last_cached_tokens)
            timings["llm_cache_write_tokens"] = float(
                self._engine.last_cache_write_tokens)
        if record_metrics:
            registry.record_turn(
                self._call_id,
                **{self._metric_prefix + k: v
                   for k, v in timings.items() if k.endswith("_ms")})
        text = self._speaker.text(token.generation_id)
        await self._append_agent(token, text, audio=True)
        self.guard.on_agent_audio_end()
        self._active_voice_generation = None

    async def _reply_text_only(self, token: GenerationToken, transcript: str) -> None:
        def is_current() -> bool:
            return self._supervisor.is_current(token)
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
                self._call_id,
                **{self._metric_prefix + "llm_ttft_ms": self._engine.last_ttft_ms})
        await self._append_agent(token, text, audio=False)

    async def _append_agent(self, token: GenerationToken, text: str,
                            audio: bool) -> None:
        if not self._supervisor.is_current(token):
            return
        self._state.conversation.append({
            "role": "agent", "text": text, "interrupted": False,
            "turn_id": token.turn_id, "generation_id": token.generation_id,
            "call_id": self._call_id,
        })
        await self._send({"type": "agent", "text": text, "interrupted": False,
                          "audio": audio, "turn_id": token.turn_id,
                          "generation_id": token.generation_id})

    async def on_script(self, text: str, turn_id: int) -> None:
        """Deliver deterministic disclosure/consent/closing copy without an LLM."""
        await self._replace_current()
        if self._speaker is None:
            await self._supervisor.start(
                turn_id, lambda token: self._append_agent(token, text, audio=False))
            return

        async def script_sentences():
            """Same chunking as engine output: the caller hears sentence one
            while the rest is still synthesizing, the transcript grows with
            her voice, and no single utterance has to carry a whole
            paragraph."""
            chunker = SentenceChunker()
            for sentence in chunker.push(text):
                yield sentence
            tail = chunker.flush()
            if tail:
                yield tail

        now = asyncio.get_running_loop().time()
        token = await self._supervisor.start(
            turn_id,
            lambda owned: self._deliver_voice(
                owned, script_sentences(),
                {"commit_t": now, "vad_stop_t": now}, record_metrics=False),
        )
        self._drained[token.generation_id] = asyncio.Event()
        self._active_voice_generation = token.generation_id

    # --- text mode (chat box drives the engine without audio) ---

    async def on_chat(self, text: str, turn_id: int) -> None:
        await self._replace_current()
        token = await self._supervisor.start(
            turn_id, lambda token: self._reply_text_only(token, text))
        self._queue_extraction(token, text)

    def _queue_extraction(self, token: GenerationToken, transcript: str) -> None:
        """Run provisional evidence extraction beside speech, in turn order.

        Only committed turns reach this method. Once committed, their evidence
        remains valid even if the caller interrupts the agent's reply, so the
        validity check is intentionally separate from GenerationSupervisor.
        """
        if not isinstance(self._engine, LlmEngine) or self._closed:
            return
        if token.turn_id in self._committed_turns:
            return
        self._committed_turns.add(token.turn_id)
        if self.interview is not None:
            self.interview.add_history("user", transcript)
        previous = self._extraction_tail

        def is_valid() -> bool:
            return not self._closed and token.turn_id in self._committed_turns

        async def run() -> None:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    if self._closed:
                        return
                except Exception:
                    # The prior runner normally contains its own failures. Keep
                    # later committed evidence moving if it failed unexpectedly.
                    pass
            if not is_valid():
                return
            try:
                results = await self._engine.extract(
                    transcript, is_valid=is_valid, turn_id=token.turn_id,
                    generation_id=token.generation_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("background evidence extraction failed")
                return
            failures = [
                {"tool_call_id": result.get("tool_call_id"),
                 "name": result.get("name"), "reason": result.get("reason")}
                for result in results
                if (not result.get("applied")
                    and not str(result.get("reason", "")).startswith("duplicate"))
            ]
            if failures and is_valid():
                await self._send({
                    "type": "tool_failures", "turn_id": token.turn_id,
                    "generation_id": token.generation_id,
                    "failures": failures,
                })

        task = asyncio.create_task(run())
        self._extraction_tail = task
        self._extraction_tasks.add(task)
        task.add_done_callback(self._extraction_tasks.discard)

    # --- barge-in ---

    async def speculate(self, transcript: str, turn_id: int) -> None:
        """Start LLM+TTS for the expected turn BEFORE the endpointer commits.
        Audio is release-gated and the supervisor cancels the generation if the
        caller resumes or the transcript changes. Speculation cannot extract or
        mutate evidence; extraction begins only after this draft is promoted."""
        if self._speaker is None or not isinstance(self._engine, LlmEngine):
            return
        if self._spec is not None:
            if _spoken_eq(self._spec["transcript"], transcript):
                return  # already speculating on effectively this turn
            await self.cancel_speculation()  # transcript grew: guess is stale
        if self._supervisor.current is not None:
            return  # a real reply is active; never preempt it on a guess
        release = asyncio.Event()
        anchors = {"commit_t": 0.0, "vad_stop_t": 0.0}
        spec = {"transcript": transcript, "release": release, "anchors": anchors}
        log.info("speculation START %r", transcript[:60])
        spec["token"] = await self._supervisor.start(
            turn_id,
            lambda owned: self._deliver_voice(
                owned, self._voiced_sentences(
                    owned, transcript, commit_gate=release),
                anchors, record_metrics=True, release=release))
        self._spec = spec

    async def cancel_speculation(self) -> None:
        spec = self._spec
        self._spec = None
        if spec is not None and self._supervisor.is_current(spec["token"]):
            log.info("speculation CANCELLED %r", spec["transcript"][:60])
            await self._supervisor.cancel_current()

    async def _replace_current(self) -> None:
        if self._spec is not None:
            await self.cancel_speculation()
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
                "call_id": self._call_id,
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

    async def on_playback_started(self, generation_id: int) -> None:
        current = self._supervisor.current
        anchor = self._browser_turn_anchors.pop(generation_id, None)
        if (anchor is None or current is None
                or current.generation_id != generation_id):
            return
        registry.record_turn(
            self._call_id,
            **{self._metric_prefix + "browser_turn_latency_ms":
               (asyncio.get_running_loop().time() - anchor) * 1000},
        )

    def _remember_browser_anchor(self, generation_id: int, vad_stop_t: float) -> None:
        self._browser_turn_anchors[generation_id] = vad_stop_t
        while len(self._browser_turn_anchors) > 4:
            del self._browser_turn_anchors[next(iter(self._browser_turn_anchors))]

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
        self._closed = True
        if self._engine_warm is not None:
            self._engine_warm.cancel()
            with suppress(asyncio.CancelledError):
                await self._engine_warm
            self._engine_warm = None
        for task in tuple(self._extraction_tasks):
            if not task.done():
                task.cancel()
        if self._extraction_tasks:
            await asyncio.gather(*tuple(self._extraction_tasks),
                                 return_exceptions=True)
        self._extraction_tasks.clear()
        self._extraction_tail = None
        await self._supervisor.close()
        self.guard.on_agent_audio_end()
        if self._prewarm is not None:
            if not self._prewarm.done():
                self._prewarm.cancel()
            try:
                await self._prewarm
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.info("tts prewarm did not complete: %s", exc)
        if self._tts is not None:
            await self._tts.close()
        if isinstance(self._engine, LlmEngine):
            await self._engine.close()
