"""Per-connection call session: audio in -> VAD + ASR -> endpointer -> replies.

## How this works
One CallSession per WebSocket, three long-lived tasks under a TaskGroup:
- _recv_loop (master): browser frames -> VAD (via asyncio.to_thread; the event
  loop never blocks on ONNX) -> bounded ASR queue (DROP-OLDEST, counted). VAD
  transitions and client JSON messages ("cleared" acks, "chat" text-mode input)
  land on one internal event queue.
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
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from server.config import Settings
from server.metrics import registry
from server.realtime.asr import AsrFinal, AsrPartial, AsrUtteranceEnd, DeepgramSession
from server.realtime.endpoint import Endpointer, TurnComplete
from server.realtime.reply import ReplyController
from server.realtime.vad import SileroVad, VadEvent, VadStream

log = logging.getLogger(__name__)

TICK_SEC = 0.05
AUDIO_QUEUE_FRAMES = 50   # 1 s of audio backlog toward ASR, then drop-oldest
EVENT_QUEUE_SIZE = 200

_shared_vad: SileroVad | None = None  # model loaded once per process, not per call


def _get_vad() -> SileroVad:
    global _shared_vad
    if _shared_vad is None:
        _shared_vad = SileroVad()
    return _shared_vad


@dataclass(frozen=True)
class ClientCleared:
    played_samples: int


@dataclass(frozen=True)
class ClientChat:
    text: str


@dataclass
class CallState:
    turns: list[TurnComplete] = field(default_factory=list)
    replies: list[dict] = field(default_factory=list)
    frames_in: int = 0
    frames_dropped: int = 0
    vad_events_dropped: int = 0


class CallSession:
    def __init__(self, ws: WebSocket, settings: Settings) -> None:
        self._ws = ws
        self._settings = settings
        self.state = CallState()
        self._audio_to_asr: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_FRAMES)
        self._events: asyncio.Queue = asyncio.Queue(maxsize=EVENT_QUEUE_SIZE)
        self._endpointer = Endpointer()

    async def run(self) -> None:
        await self._ws.accept()
        if not self._settings.deepgram_api_key:
            await self._send({"type": "error", "message": "DEEPGRAM_API_KEY is not set"})
            await self._ws.close()
            return
        replies = ReplyController(self._send, self._ws.send_bytes,
                                  self._settings, self.state)
        self._replies = replies
        vad_model = _get_vad()
        vad_model.reset()
        self._vad = VadStream(vad=vad_model)
        asr = DeepgramSession(self._settings.deepgram_api_key, self._events)
        try:
            async with asyncio.TaskGroup() as tg:
                recv = tg.create_task(self._recv_loop())
                tg.create_task(asr.run(self._audio_to_asr))
                events = tg.create_task(self._event_loop())
                await recv
                events.cancel()
        except* WebSocketDisconnect:
            pass
        finally:
            await replies.close()
            log.info(
                "call ended: %d turns, %d replies, %d frames in, %d dropped",
                len(self.state.turns), len(self.state.replies),
                self.state.frames_in, self.state.frames_dropped,
            )

    async def _send(self, payload: dict) -> None:
        await self._ws.send_text(json.dumps(payload))

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
        match msg.get("type"):
            case "cleared":
                self._offer_event(ClientCleared(int(msg.get("played_samples", 0))))
            case "chat":
                chat_text = str(msg.get("text", "")).strip()
                if chat_text:
                    self._offer_event(ClientChat(chat_text))

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
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            self.state.vad_events_dropped += 1
            log.error("pipeline event dropped — event queue full")

    # --- pipeline events -> endpointer + guard -> replies ---

    async def _event_loop(self) -> None:
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
                    await self._send({"type": "vad", "state": "speech"})
                case VadEvent(kind="stop", t=t):
                    self._endpointer.on_vad_stop(t)
                    self._replies.guard.on_vad_stop(t)
                    await self._send({"type": "vad", "state": "silence"})
                case AsrPartial(text=text):
                    await self._send({"type": "partial", "text": text})
                case AsrFinal(text=text):
                    self._endpointer.on_asr_final(text, now)
                    if text:
                        await self._send({"type": "final", "text": text})
                case AsrUtteranceEnd():
                    turn = self._endpointer.on_utterance_end(now)
                case ClientCleared(played_samples=played):
                    await self._replies.on_cleared(played)
                case ClientChat(text=text):
                    await self._send({"type": "you", "text": text})
                    await self._replies.on_chat(text)
                case None:
                    pass
            turn = turn or self._endpointer.tick(now)
            if turn is not None:
                self.state.turns.append(turn)
                registry.record_turn(endpoint_delay_ms=turn.endpoint_delay * 1000)
                await self._send({
                    "type": "turn",
                    "transcript": turn.transcript,
                    "endpoint_delay_ms": round(turn.endpoint_delay * 1000),
                    "reason": turn.reason,
                })
                await self._replies.on_turn(turn)
            if self._replies.guard.tick(now):
                self._replies.cancel_current()
                await self._send({"type": "clear"})
