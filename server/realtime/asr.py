"""Deepgram v1 streaming ASR client (raw WebSocket, no SDK). Phase 2.

## How this works
We speak Deepgram's wire protocol directly: one wss connection to /v1/listen with
everything configured in the query string (nova-3, linear16/16k; endpointing=300
gives us speech_final flags, utterance_end_ms=1000 gives UtteranceEnd events —
the latter REQUIRES interim_results=true because it works by scanning interim
word gaps). Binary frames go up, JSON events come down.

DeepgramSession.run() drives two loops under a TaskGroup, so if either dies the
other is cancelled and the context manager closes the socket (cancellation-
correct by construction):
- _send_loop pulls frames from the audio queue. If nothing arrives for 4 s it
  sends {"type":"KeepAlive"} — Deepgram closes idle connections after 10 s. A
  None sentinel in the queue means end-of-call: send {"type":"CloseStream"} so
  Deepgram flushes its final transcripts before we hang up.
- _recv_loop parses each message into a typed event and pushes it to events_out
  (the EventBuffer: stale partials are replaceable under pressure; finals and
  control events are never dropped — see events.py). Never blocks the reader.
run() retries ONCE on an abnormal close (each attempt is a connection epoch; a
reconnect emits AsrReconnected so the consumer can surface it), then fails with
the typed AsrUnavailable. Shutdown is bounded: after CloseStream the receiver
gets SHUTDOWN_TIMEOUT_SEC to drain finals before being cancelled — a hung
provider cannot hold call teardown hostage.

Design note (interview): this is Deepgram v1 + our own endpointer (endpoint.py)
by deliberate choice. Deepgram's newer Flux model (v2/listen) has model-integrated
end-of-turn detection that subsumes this file's downstream policy — the managed
alternative we'd evaluate in production; here, owning the turn-taking policy is
the point.
"""
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

log = logging.getLogger(__name__)

KEEPALIVE_IDLE_SEC = 4  # Deepgram closes after 10 s of silence; stay well under
CONNECT_TIMEOUT_SEC = 10   # bound the handshake: fail typed, not hung
SHUTDOWN_TIMEOUT_SEC = 5   # after CloseStream: how long we wait for final flush


class AsrUnavailable(RuntimeError):
    """The ASR provider could not be reached or kept failing after a retry."""

FINALIZE = "finalize"  # audio-queue control marker -> {"type":"Finalize"}.
# Why: Deepgram's own endpointing finalizes short utterances ("Five.") only
# after 1.5-3.5 s. Our local VAD knows the caller stopped ~100 ms in, so on
# vad_stop we queue this marker and Deepgram flushes finals immediately —
# measured endpoint_delay drops from ~1.5 s p50 to the fast-timer floor.


@dataclass(frozen=True)
class AsrPartial:
    text: str


@dataclass(frozen=True)
class AsrFinal:
    text: str
    speech_final: bool


@dataclass(frozen=True)
class AsrUtteranceEnd:
    pass


@dataclass(frozen=True)
class AsrReconnected:
    """The stream restarted on a new connection epoch. Continuity policy:
    accumulated endpointer finals survive; the next partial supersedes any
    stale one (partials are replaceable by design — see events.py)."""
    epoch: int


AsrEvent = AsrPartial | AsrFinal | AsrUtteranceEnd | AsrReconnected


def build_url() -> str:
    params = {
        "model": "nova-3",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "interim_results": "true",
        "endpointing": "300",
        "utterance_end_ms": "1000",
        "punctuate": "true",
        # Transcribe "um"/"uh" instead of dropping them: hesitations are the
        # endpointer's trailing-word evidence (found by the simulated caller —
        # a dropped "um" let the fast tier split a mid-thought pause).
        "filler_words": "true",
    }
    return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"


def parse_message(raw: str) -> AsrEvent | None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    match msg.get("type"):
        case "Results":
            alts = msg.get("channel", {}).get("alternatives", [])
            text = alts[0].get("transcript", "") if alts else ""
            if msg.get("is_final"):
                return AsrFinal(text=text, speech_final=bool(msg.get("speech_final")))
            return AsrPartial(text=text) if text else None
        case "UtteranceEnd":
            return AsrUtteranceEnd()
        case _:
            return None


class DeepgramSession:
    def __init__(self, api_key: str, events_out: asyncio.Queue, url: str | None = None) -> None:
        self._api_key = api_key
        self._events_out = events_out
        self._url = url or build_url()
        self.dropped_events = 0

    async def run(self, audio: asyncio.Queue) -> None:
        for attempt in (1, 2):  # reconnect-once policy, each attempt = an epoch
            try:
                await self._run_once(audio)
                return
            except (websockets.ConnectionClosedError, OSError, TimeoutError) as exc:
                log.warning("deepgram connection lost (attempt %d): %s", attempt, exc)
                if attempt == 2:
                    raise AsrUnavailable("speech recognition unavailable") from exc
                self._events_out.put_nowait(AsrReconnected(epoch=attempt + 1))

    async def _run_once(self, audio: asyncio.Queue) -> None:
        async with websockets.connect(
            self._url, additional_headers={"Authorization": f"Token {self._api_key}"},
            open_timeout=CONNECT_TIMEOUT_SEC,
        ) as ws:
            await self._pump(ws, audio)

    async def _pump(self, ws, audio: asyncio.Queue) -> None:
        """Sender drives; after the CloseStream sentinel the receiver gets a
        bounded window to drain finals — a hung provider cannot hold teardown."""
        recv = asyncio.create_task(self._recv_loop(ws))
        try:
            await self._send_loop(ws, audio)
            try:
                await asyncio.wait_for(asyncio.shield(recv), SHUTDOWN_TIMEOUT_SEC)
            except TimeoutError:
                log.warning("asr shutdown flush timed out; forcing close")
        finally:
            if not recv.done():
                recv.cancel()
            with suppress(asyncio.CancelledError):
                await recv

    async def _send_loop(self, ws, audio: asyncio.Queue) -> None:
        while True:
            try:
                frame = await asyncio.wait_for(audio.get(), timeout=KEEPALIVE_IDLE_SEC)
            except TimeoutError:
                await ws.send(json.dumps({"type": "KeepAlive"}))
                continue
            if frame is None:  # end-of-call sentinel
                await ws.send(json.dumps({"type": "CloseStream"}))
                return
            if frame is FINALIZE:  # VAD said speech stopped: flush finals now
                await ws.send(json.dumps({"type": "Finalize"}))
                continue
            await ws.send(frame)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            event = parse_message(raw)
            if event is None:
                continue
            try:
                self._events_out.put_nowait(event)
            except asyncio.QueueFull:
                # Drop-newest: transcripts refresh continuously, so a fresher one
                # follows; blocking here would back-pressure the socket reader.
                self.dropped_events += 1
