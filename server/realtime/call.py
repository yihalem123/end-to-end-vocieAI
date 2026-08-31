"""Per-connection call session: audio in -> VAD + ASR -> endpointer -> replies.

## How this works
One CallSession per WebSocket, three long-lived tasks under a TaskGroup:
- _recv_loop (master): browser frames -> VAD (via asyncio.to_thread; the event
  loop never blocks on ONNX) -> bounded ASR audio queue (DROP-OLDEST, counted).
  VAD transitions and client JSON acks land on the EventBuffer, where stale
  partials are replaceable and finals/acks are never dropped (events.py).
- DeepgramSession.run: audio queue -> Deepgram -> typed ASR events, same queue.
- _event_loop: single consumer. Feeds the Endpointer and ticks the barge-in
  guard (sync state machines + real time). Committed turns and chat messages go
  to the ReplyController (reply.py), which owns engine/TTS/barge-in mechanics.
Cancellation: client disconnect ends _recv_loop -> ASR gets its CloseStream
sentinel, the event loop is cancelled, the controller closes engine and TTS —
every task cleans up in finally/context managers (CLAUDE.md cancellation rules).
"""
import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from server.config import Settings
from server.engine.intents import EndCallIntent, classify_end_call_intent
from server.metrics import registry
from server.realtime.asr import (
    FINALIZE,
    AsrFinal,
    AsrPartial,
    AsrReconnected,
    AsrUnavailable,
    AsrUtteranceEnd,
    DeepgramSession,
)
from server.realtime.endpoint import Endpointer, TurnComplete
from server.engine.plan import InterviewPlan
from server.realtime.events import CriticalEventOverflow, EventBuffer
from server.realtime.flux import (
    FluxEagerEndOfTurn,
    FluxEndOfTurn,
    FluxSession,
    FluxStartOfTurn,
    FluxTurnResumed,
    FluxUpdate,
)
from server.realtime.reply import ReplyController
from server.realtime.session import SessionLifecycle, SessionStatus, classify_consent
from server.realtime.vad import SileroRuntime, SileroVad, VadEvent, VadStream

log = logging.getLogger(__name__)

TICK_SEC = 0.025
# 5 s of audio toward ASR before drop-oldest kicks in. Sized for STARTUP, not
# steady state: the first utterance races the Deepgram connect handshake
# (~0.5-1.5 s), and a 1 s queue ate the caller's greeting (found by the
# simulated caller: VAD fired, zero transcripts, 54 drops). 5 s costs 160 KB.
AUDIO_QUEUE_FRAMES = 250
EVENT_RELIABLE_LIMIT = 256
PARTIAL_UI_INTERVAL_SEC = 0.05


@dataclass(frozen=True)
class ClientCleared:
    generation_id: int
    played_samples: int


@dataclass(frozen=True)
class ClientPlaybackDrained:
    generation_id: int


@dataclass(frozen=True)
class ClientPlaybackStarted:
    generation_id: int


@dataclass(frozen=True)
class ClientPlaybackOverflow:
    generation_id: int
    played_samples: int


@dataclass(frozen=True)
class ClientChat:
    text: str


@dataclass
class CallState:
    call_id: str
    session: SessionLifecycle
    turns: list[TurnComplete] = field(default_factory=list)
    # Ordered conversation log — the post-call transcript source of truth.
    # Entries: {"role": "caller"|"agent", "text": ..., "interrupted": bool?}
    conversation: list[dict] = field(default_factory=list)
    frames_in: int = 0
    frames_dropped: int = 0
    vad_events_dropped: int = 0
    stale_asr_events: int = 0


class CallSession:
    def __init__(self, ws: WebSocket, settings: Settings, call_id: str,
                 mode: str = "custom", plan: InterviewPlan | None = None,
                 vad_runtime: SileroRuntime | None = None) -> None:
        self._ws = ws
        self._settings = settings
        # "custom": v1 ASR + our VAD-anchored endpointer (the built story).
        # "flux": v2 model end-of-turn — EndOfTurn IS the commit; the
        # endpointer is bypassed. Local VAD runs in BOTH modes (barge-in
        # needs ~100ms local onset detection; network events are too late).
        self.mode = mode if mode in ("custom", "flux") else "custom"
        self._metric_prefix = "flux_" if self.mode == "flux" else ""
        self.state = CallState(call_id=call_id, session=SessionLifecycle(call_id))
        self._audio_to_asr: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_FRAMES)
        self._events = EventBuffer(reliable_limit=EVENT_RELIABLE_LIMIT)
        self._endpointer = Endpointer()
        self._plan = plan
        self._vad_runtime = vad_runtime
        self._asr_epoch = 1
        self._last_partial_sent_t = 0.0
        self._last_vad_stop_t: float | None = None
        self._next_turn_id = 0
        self._close_when_idle = False
        self._pending_end_confirmation = False
        self._interview_started_t: float | None = None

    async def run(self) -> None:
        await self._ws.accept()
        await self._send({"type": "session", "call_id": self.state.call_id,
                          "state": self.state.session.status})
        if not self._settings.deepgram_api_key:
            self.state.session.transition(SessionStatus.FAILED)
            await self._send_state()
            await self._send({"type": "error", "message": "DEEPGRAM_API_KEY is not set"})
            await self._ws.close()
            return
        replies = ReplyController(self._send, self._ws.send_bytes,
                                  self._settings, self.state,
                                  metric_prefix=self._metric_prefix,
                                  call_id=self.state.call_id, plan=self._plan)
        self._replies = replies
        # Only immutable ONNX resources are shared. Recurrent state, context,
        # carry, gate and reset lifetime belong exclusively to this call.
        runtime = self._vad_runtime
        if runtime is None:
            runtime = await asyncio.to_thread(SileroRuntime)
        self._vad = VadStream(vad=SileroVad(runtime))
        if self.mode == "flux":
            from server.realtime.flux import build_flux_url
            asr = FluxSession(
                self._settings.deepgram_api_key,
                self._events,
                url=build_flux_url(self._settings.flux_eot_threshold,
                                   self._settings.flux_eager_eot_threshold),
            )
        else:
            asr = DeepgramSession(self._settings.deepgram_api_key, self._events)
        log.info("call starting in %s mode", self.mode)
        try:
            async with asyncio.TaskGroup() as tg:
                recv = tg.create_task(self._recv_loop())
                tg.create_task(asr.run(self._audio_to_asr))
                events = tg.create_task(self._event_loop())
                await recv
                events.cancel()
        except* WebSocketDisconnect:
            pass
        except* AsrUnavailable:
            # Typed provider failure: tell the caller and end cleanly instead
            # of surfacing a bare traceback through the app layer.
            log.error("asr unavailable; ending call %s", self.state.call_id)
            self.state.session.transition(SessionStatus.FAILED)
            with suppress(Exception):
                await self._send({"type": "error",
                                  "message": "speech recognition unavailable"})
                await self._ws.close(code=1011)
        except* CriticalEventOverflow:
            log.error("reliable event lane overflow; ending call %s",
                      self.state.call_id)
            self.state.session.transition(SessionStatus.FAILED)
            with suppress(Exception):
                await self._send({"type": "error",
                                  "message": "realtime pipeline overloaded"})
                await self._ws.close(code=1011)
        finally:
            await replies.close()
            log.info(
                "call ended: %d turns, %d conversation entries, %d frames in, "
                "%d dropped, %d stale partials replaced, %d stale ASR events",
                len(self.state.turns), len(self.state.conversation),
                self.state.frames_in, self.state.frames_dropped,
                self._events.replaced,
                self.state.stale_asr_events,
            )

    async def _send(self, payload: dict) -> None:
        payload.setdefault("call_id", self.state.call_id)
        await self._ws.send_text(json.dumps(payload))

    async def _send_state(self) -> None:
        await self._send({"type": "session_state",
                          "state": self.state.session.status})

    # --- inbound: audio frames and client JSON ---

    async def _recv_loop(self) -> None:
        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("text") is not None:
                    self._handle_client_json(message["text"])
                    continue
                frame = message.get("bytes")
                if frame is None:
                    continue
                now = time.monotonic()
                self.state.frames_in += 1
                for event in await asyncio.to_thread(self._vad.feed, frame, now):
                    self._offer_event(event)
                self._offer_audio(frame)
        finally:
            self._offer_audio(None, force=True)  # ASR must always get CloseStream

    def _handle_client_json(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        try:
            match msg.get("type"):
                case "cleared":
                    generation = int(msg.get("generation_id", 0))
                    if generation > 0:
                        self._offer_event(ClientCleared(
                            generation, max(0, int(msg.get("played_samples", 0)))))
                case "playback_drained":
                    generation = int(msg.get("generation_id", 0))
                    if generation > 0:
                        self._offer_event(ClientPlaybackDrained(generation))
                case "playback_started":
                    generation = int(msg.get("generation_id", 0))
                    if generation > 0:
                        self._offer_event(ClientPlaybackStarted(generation))
                case "playback_overflow":
                    generation = int(msg.get("generation_id", 0))
                    if generation > 0:
                        self._offer_event(ClientPlaybackOverflow(
                            generation, max(0, int(msg.get("played_samples", 0)))))
                case "chat":
                    chat_text = str(msg.get("text", "")).strip()
                    if chat_text:
                        self._offer_event(ClientChat(chat_text))
        except (TypeError, ValueError):
            return  # malformed client control messages never kill the call

    def _offer_audio(self, frame: bytes | None, force: bool = False) -> None:
        try:
            self._audio_to_asr.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop-OLDEST: behind on ASR, the oldest frame is the least useful.
            try:
                self._audio_to_asr.get_nowait()
                if not force:
                    self.state.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass
            self._audio_to_asr.put_nowait(frame)

    def _offer_event(self, event) -> None:
        # Partials replace in one slot. Reliable overflow raises loudly: losing
        # a final/VAD/ack would be worse than ending the affected call.
        self._events.put_nowait(event)

    def _accept_asr_epoch(self, epoch: int) -> bool:
        if epoch == self._asr_epoch:
            return True
        self.state.stale_asr_events += 1
        log.warning("discarding stale ASR event epoch=%d current=%d call=%s",
                    epoch, self._asr_epoch, self.state.call_id)
        return False

    @property
    def tool_ledger(self) -> list[dict]:
        """Audit trail of the engine's tool calls for the post-call report."""
        interview = getattr(getattr(self, "_replies", None), "interview", None)
        return list(getattr(interview, "tool_ledger", []) or [])

    def _new_turn_id(self) -> int:
        self._next_turn_id += 1
        return self._next_turn_id

    # --- pipeline events -> endpointer + guard -> replies ---

    async def _event_loop(self) -> None:
        await self._replies.on_script(self._replies.interview.plan.consent, turn_id=0)
        self.state.session.transition(SessionStatus.AWAITING_CONSENT)
        await self._send_state()
        while True:
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=TICK_SEC)
            except TimeoutError:
                event = None
            now = time.monotonic()
            started = getattr(self, "_interview_started_t", None)
            if started is not None:
                self._replies.interview.update_elapsed(now - started)
            turn: TurnComplete | None = None
            match event:
                case VadEvent(kind="start", t=t):
                    self._endpointer.on_vad_start(t)
                    self._replies.guard.on_vad_start(t)
                    await self._replies.cancel_speculation()  # guess voided
                    await self._send({"type": "vad", "state": "speech"})
                case VadEvent(kind="stop", t=t):
                    self._endpointer.on_vad_stop(t)
                    self._replies.guard.on_vad_stop(t)
                    self._last_vad_stop_t = t
                    if self.mode == "custom":
                        self._offer_audio(FINALIZE)  # flush v1 finals now
                    await self._send({"type": "vad", "state": "silence"})
                case AsrPartial(text=text, epoch=epoch) if self._accept_asr_epoch(epoch):
                    if now - self._last_partial_sent_t >= PARTIAL_UI_INTERVAL_SEC:
                        self._last_partial_sent_t = now
                        await self._send({"type": "partial", "text": text})
                case AsrFinal(text=text, epoch=epoch) if self._accept_asr_epoch(epoch):
                    self._endpointer.on_asr_final(text, now)
                    if text:
                        await self._send({"type": "final", "text": text})
                case AsrUtteranceEnd(epoch=epoch) if self._accept_asr_epoch(epoch):
                    turn = self._endpointer.on_utterance_end(now)
                case AsrReconnected(epoch=epoch):
                    # Continuity: accumulated finals survive; next partial
                    # supersedes any stale one. The caller sees a brief notice.
                    if epoch <= self._asr_epoch:
                        continue
                    self._asr_epoch = epoch
                    log.warning("asr reconnected (epoch %d) call %s",
                                epoch, self.state.call_id)
                    await self._send({"type": "notice",
                                      "text": "transcription reconnected"})
                case ClientCleared(generation_id=generation, played_samples=played):
                    await self._replies.on_cleared(generation, played)
                case ClientPlaybackDrained(generation_id=generation):
                    await self._replies.on_playback_drained(generation)
                case ClientPlaybackStarted(generation_id=generation):
                    await self._replies.on_playback_started(generation)
                case ClientPlaybackOverflow(
                    generation_id=generation, played_samples=played
                ):
                    await self._replies.on_playback_overflow(generation, played)
                case ClientChat(text=text):
                    turn_id = self._new_turn_id()
                    await self._commit_caller_text(text, turn_id, turn=None)
                case FluxUpdate(transcript=text, epoch=epoch) if self._accept_asr_epoch(epoch):
                    if text:  # Flux emits empty updates during silence
                        if now - self._last_partial_sent_t >= PARTIAL_UI_INTERVAL_SEC:
                            self._last_partial_sent_t = now
                            await self._send({"type": "partial", "text": text})
                case FluxEagerEndOfTurn(transcript=text, epoch=epoch) \
                        if self._accept_asr_epoch(epoch):
                    if (text and self.state.session.status
                            == SessionStatus.INTERVIEWING):
                        await self._replies.speculate(text, self._next_turn_id + 1)
                case FluxTurnResumed(epoch=epoch) if self._accept_asr_epoch(epoch):
                    await self._replies.cancel_speculation()
                case FluxEndOfTurn(transcript=text, epoch=epoch) \
                        if self._accept_asr_epoch(epoch):
                    if text:  # empty end-of-turn = silence, not a turn
                        anchor = self._last_vad_stop_t or now
                        turn = TurnComplete(
                            transcript=text, endpoint_delay=max(0.0, now - anchor),
                            vad_stop_t=anchor, commit_t=now, reason="flux")
                case FluxStartOfTurn(epoch=epoch) if self._accept_asr_epoch(epoch):
                    pass  # local VAD already drives the UI + barge-in guard
                case None:
                    pass
            if self.mode == "custom":
                turn = turn or self._endpointer.tick(now)
            if turn is not None:
                turn_id = self._new_turn_id()
                self.state.turns.append(turn)
                registry.record_turn(
                    self.state.call_id,
                    **{f"{self._metric_prefix}endpoint_delay_ms":
                       turn.endpoint_delay * 1000})
                await self._commit_caller_text(turn.transcript, turn_id, turn=turn)
            if (turn is None and self.mode == "custom"
                    and self.state.session.status == SessionStatus.INTERVIEWING
                    and self._endpointer.pending_complete):
                # Final-seeded speculation only: interim text keeps trickling in
                # for ~400 ms after silence, so partial-seeded guesses churn
                # (measured: cancel/restart per partial, zero net gain). Once
                # the punctuated final lands the text is stable and the head
                # start is real. Audio stays commit-gated either way.
                await self._replies.speculate(
                    self._endpointer.transcript, self._next_turn_id + 1)
            if self._replies.guard.tick(now):
                await self._replies.interrupt_current()
            if (self.state.session.status == SessionStatus.INTERVIEWING
                    and self._replies.is_idle):
                request = self._replies.interview.end_call_request
                if request is not None:
                    self.state.session.transition(SessionStatus.CLOSING)
                    await self._send_state()
                    self._close_when_idle = True
                elif self._replies.interview.done:
                    # Speech already responded to the final answer. Extraction
                    # completing later must close lifecycle state without
                    # injecting a second, duplicate goodbye.
                    self._replies.interview.request_end_call(
                        "interview_complete",
                        "The interview objectives were completed.",
                    )
                    self.state.session.transition(SessionStatus.CLOSING)
                    await self._send_state()
                    self._close_when_idle = True
                elif (self._replies.interview.elapsed_seconds >=
                      self._replies.interview.plan.boundaries.max_duration_minutes * 60):
                    await self._finish_interview(
                        "max_duration",
                        "We've reached the time limit, so I'll end the screening here. Thank you.",
                        self._next_turn_id,
                    )
                elif (self._replies.interview.caller_turn_count >=
                      self._replies.interview.plan.boundaries.max_turns):
                    await self._finish_interview(
                        "max_turns",
                        "We've reached the interview limit, so I'll end the screening here. Thank you.",
                        self._next_turn_id,
                    )
            if self._close_when_idle and self._replies.is_idle:
                await self._ws.close(code=1000)
                return

    async def _commit_caller_text(self, text: str, turn_id: int,
                                  turn: TurnComplete | None) -> None:
        utterance_id = f"{self.state.call_id}:u{turn_id}"
        self.state.conversation.append({
            "role": "caller", "text": text, "turn_id": turn_id,
            "utterance_id": utterance_id, "call_id": self.state.call_id,
        })
        if turn is None:
            await self._send({"type": "you", "text": text, "turn_id": turn_id,
                              "utterance_id": utterance_id})
        else:
            await self._send({
                "type": "turn", "turn_id": turn_id,
                "utterance_id": utterance_id, "transcript": text,
                "endpoint_delay_ms": round(turn.endpoint_delay * 1000),
                "reason": turn.reason,
            })

        status = self.state.session.status
        if status == SessionStatus.AWAITING_CONSENT:
            consent = classify_consent(text)
            if consent is None:
                await self._replies.on_script(
                    "I need a clear yes or no before continuing. Do you consent?",
                    turn_id,
                )
                return
            self._replies.interview.record("consent", consent, quote=text)
            if not consent:
                self.state.session.transition(SessionStatus.CONSENT_REFUSED)
                await self._send_state()
                await self._replies.on_script(
                    "Understood. I will not continue the screening. Thank you.",
                    turn_id,
                )
                self._close_when_idle = True
                return
            self.state.session.transition(SessionStatus.INTERVIEWING)
            self._interview_started_t = time.monotonic()
            await self._send_state()
        elif status != SessionStatus.INTERVIEWING:
            return

        if self.state.session.status == SessionStatus.INTERVIEWING:
            self._replies.interview.note_caller_turn()
            started = getattr(self, "_interview_started_t", None)
            if started is not None:
                self._replies.interview.update_elapsed(time.monotonic() - started)
            if getattr(self, "_pending_end_confirmation", False):
                confirmation = classify_consent(text)
                repeated_end_intent = classify_end_call_intent(text)
                if (confirmation is True
                        or repeated_end_intent in {
                            EndCallIntent.END, EndCallIntent.CONFIRM}):
                    self._pending_end_confirmation = False
                    await self._finish_interview(
                        "candidate_requested",
                        "Understood. I'll end the call here. Thank you for your time.",
                        turn_id,
                    )
                    return
                if confirmation is False:
                    self._pending_end_confirmation = False
                    await self._replies.on_script(
                        "Okay, we can continue.", turn_id)
                    return
                await self._replies.on_script(
                    "Please say yes if you want me to end the call, or no to continue.",
                    turn_id,
                )
                return

            intent = classify_end_call_intent(text)
            if intent == EndCallIntent.END:
                await self._finish_interview(
                    "candidate_requested",
                    "Understood. I'll end the call here. Thank you for your time.",
                    turn_id,
                )
                return
            if intent == EndCallIntent.CONFIRM:
                self._pending_end_confirmation = True
                await self._replies.on_script(
                    "Did you ask me to end the call? Please say yes or no.",
                    turn_id,
                )
                return

        if turn is None:
            await self._replies.on_chat(text, turn_id)
        else:
            await self._replies.on_turn(turn, turn_id)

    async def _finish_interview(self, reason: str, closing_message: str,
                                turn_id: int) -> None:
        """One validated owner for graceful termination and closing playback."""
        self._replies.interview.request_end_call(reason, closing_message)
        if self.state.session.status == SessionStatus.INTERVIEWING:
            self.state.session.transition(SessionStatus.CLOSING)
            await self._send_state()
        await self._replies.on_script(closing_message, turn_id)
        self._close_when_idle = True
