"""Per-connection call session: audio in -> VAD + ASR -> endpointer -> events out.

## How this works
One CallSession per WebSocket connection, three tasks under a TaskGroup:
- _recv_loop (the master): reads 20 ms frames from the browser. Each frame goes
  (a) through VAD — inference runs via asyncio.to_thread per CLAUDE.md so the
  event loop never blocks on ONNX — and (b) into the bounded ASR queue with a
  DROP-OLDEST policy (stale audio is worthless once we're behind; every drop is
  counted, never silent). VAD transitions become events on the same internal
  event queue the ASR events land on.
- DeepgramSession.run: pumps that audio queue to Deepgram, pushes typed ASR
  events back (see asr.py).
- _event_loop: THE single consumer and single ws-sender. It merges VAD + ASR
  events into the Endpointer, tick()s it every 50 ms with real time (the
  endpointer itself is a sync state machine — see endpoint.py), and streams
  UI events (vad / partial / final / turn) to the browser as JSON text. One
  sender task means no interleaved-write races on the socket.
Client disconnect ends _recv_loop, which sends the ASR sentinel (CloseStream)
and cancels the group; each task cleans up in its own finally / context manager,
so cancellation leaves no dangling socket to Deepgram.
CallState collects the per-turn timestamps (vad_stop, endpoint_commit) and drop
counters — the raw material for /metrics in Phase 3.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from server.config import Settings
from server.realtime.asr import AsrFinal, AsrPartial, AsrUtteranceEnd, DeepgramSession
from server.realtime.endpoint import Endpointer, TurnComplete
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


@dataclass
class CallState:
    turns: list[TurnComplete] = field(default_factory=list)
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
        # NOTE: VadStream state (carry buffer, gate) is per-call; only the ONNX
        # session is shared. Model state resets so call N doesn't hear call N-1.
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
                events.cancel()  # client is gone; stop the sender
                # asr task ends on its own: recv_loop queued the None sentinel.
        except* WebSocketDisconnect:
            pass
        finally:
            log.info(
                "call ended: %d turns, %d frames in, %d dropped",
                len(self.state.turns), self.state.frames_in, self.state.frames_dropped,
            )

    async def _send(self, payload: dict) -> None:
        await self._ws.send_text(json.dumps(payload))

    # --- inbound audio ---

    async def _recv_loop(self) -> None:
        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                frame = message.get("bytes")
                if frame is None:
                    continue  # text from client: unused in Phase 2
                now = time.monotonic()
                self.state.frames_in += 1
                for event in await asyncio.to_thread(self._vad.feed, frame, now):
                    self._offer_event(event)
                self._offer_audio(frame)
        finally:
            # Always tell ASR the stream is over, even on abrupt cancellation,
            # so Deepgram flushes finals / closes instead of idling to timeout.
            self._offer_audio(None, force=True)

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

    def _offer_event(self, event: VadEvent) -> None:
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            # Should not happen (consumer is fast); losing a VAD transition
            # desyncs the endpointer, so it is counted loudly.
            self.state.vad_events_dropped += 1
            log.error("vad event dropped — event queue full")

    # --- pipeline events -> endpointer -> client ---

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
                    await self._send({"type": "vad", "state": "speech"})
                case VadEvent(kind="stop", t=t):
                    self._endpointer.on_vad_stop(t)
                    await self._send({"type": "vad", "state": "silence"})
                case AsrPartial(text=text):
                    await self._send({"type": "partial", "text": text})
                case AsrFinal(text=text):
                    self._endpointer.on_asr_final(text, now)
                    if text:
                        await self._send({"type": "final", "text": text})
                case AsrUtteranceEnd():
                    turn = self._endpointer.on_utterance_end(now)
                case None:
                    pass
            turn = turn or self._endpointer.tick(now)
            if turn is not None:
                self.state.turns.append(turn)
                await self._send({
                    "type": "turn",
                    "transcript": turn.transcript,
                    "endpoint_delay_ms": round(turn.endpoint_delay * 1000),
                    "reason": turn.reason,
                })
