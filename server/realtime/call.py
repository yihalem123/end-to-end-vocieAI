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
from server.realtime.events import EventBuffer
from server.realtime.flux import FluxEndOfTurn, FluxSession, FluxStartOfTurn, FluxUpdate
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
EVENT_REPLACEABLE_LIMIT = 64  # stale partials evictable; finals/acks never drop

_shared_vad_runtime: SileroRuntime | None = None  # immutable model resources only


def _get_vad_runtime() -> SileroRuntime:
    global _shared_vad_runtime
    if _shared_vad_runtime is None:
        _shared_vad_runtime = SileroRuntime()
    return _shared_vad_runtime


@dataclass(frozen=True)
class ClientCleared:
    generation_id: int
    played_samples: int


@dataclass(frozen=True)
class ClientPlaybackDrained:
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


class CallSession:
    def __init__(self, ws: WebSocket, settings: Settings, call_id: str,
                 mode: str = "custom") -> None:
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
        self._events = EventBuffer(replaceable_limit=EVENT_REPLACEABLE_LIMIT)
        self._endpointer = Endpointer()
        self._last_vad_stop_t: float | None = None
        self._next_turn_id = 0
        self._close_when_idle = False

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
                                  call_id=self.state.call_id)
        self._replies = replies
        # Only immutable ONNX resources are shared. Recurrent state, context,
        # carry, gate and reset lifetime belong exclusively to this call.
        self._vad = VadStream(vad=SileroVad(_get_vad_runtime()))
        if self.mode == "flux":
            from server.realtime.flux import build_flux_url
            asr = FluxSession(self._settings.deepgram_api_key, self._events,
                              url=build_flux_url(self._settings.flux_eot_threshold))
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
        finally:
            await replies.close()
            log.info(
                "call ended: %d turns, %d conversation entries, %d frames in, "
                "%d dropped, %d stale partials replaced",
                len(self.state.turns), len(self.state.conversation),
                self.state.frames_in, self.state.frames_dropped,
                self._events.replaced,
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
        # EventBuffer never raises: stale partials are evicted under pressure,
        # finals/VAD/client acks are always admitted (see events.py).
        self._events.put_nowait(event)

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
                case AsrPartial(text=text):
                    await self._send({"type": "partial", "text": text})
                case AsrFinal(text=text):
                    self._endpointer.on_asr_final(text, now)
                    if text:
                        await self._send({"type": "final", "text": text})
                case AsrUtteranceEnd():
                    turn = self._endpointer.on_utterance_end(now)
                case AsrReconnected(epoch=epoch):
                    # Continuity: accumulated finals survive; next partial
                    # supersedes any stale one. The caller sees a brief notice.
                    log.warning("asr reconnected (epoch %d) call %s",
                                epoch, self.state.call_id)
                    await self._send({"type": "notice",
                                      "text": "transcription reconnected"})
                case ClientCleared(generation_id=generation, played_samples=played):
                    await self._replies.on_cleared(generation, played)
                case ClientPlaybackDrained(generation_id=generation):
                    await self._replies.on_playback_drained(generation)
                case ClientPlaybackOverflow(
                    generation_id=generation, played_samples=played
                ):
                    await self._replies.on_playback_overflow(generation, played)
                case ClientChat(text=text):
                    turn_id = self._new_turn_id()
                    await self._commit_caller_text(text, turn_id, turn=None)
                case FluxUpdate(transcript=text):
                    if text:  # Flux emits empty updates during silence
                        await self._send({"type": "partial", "text": text})
                case FluxEndOfTurn(transcript=text):
                    if text:  # empty end-of-turn = silence, not a turn
                        anchor = self._last_vad_stop_t or now
                        turn = TurnComplete(
                            transcript=text, endpoint_delay=max(0.0, now - anchor),
                            vad_stop_t=anchor, commit_t=now, reason="flux")
                case FluxStartOfTurn():
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
                    and self._replies.interview.next_needed is None
                    and self._replies.is_idle):
                self.state.session.transition(SessionStatus.CLOSING)
                await self._send_state()
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
            await self._send_state()
        elif status != SessionStatus.INTERVIEWING:
            return

        if turn is None:
            await self._replies.on_chat(text, turn_id)
        else:
            await self._replies.on_turn(turn, turn_id)
