"""Deepgram Flux client (v2/listen): model-integrated end-of-turn. Phase 5c.

## How this works
Flux is the road we deliberately didn't take in Phase 2, now built as a live
A/B: a conversational STT model whose turn-taking is acoustic + semantic and
lives INSIDE the model. One wss connection to /v2/listen; binary frames go up;
TurnInfo events come down:
  StartOfTurn  -> the caller began speaking (UI signal; barge-in stays local VAD)
  Update       -> in-turn partial transcript
  EndOfTurn    -> the model says the turn is over, transcript final
In Flux mode the custom endpointer is bypassed entirely — EndOfTurn IS the
commit. eot_threshold (0.7 default) is the one knob: higher = more patient.
EagerEndOfTurn starts a commit-gated draft; TurnResumed cancels it and EndOfTurn
promotes only the matching transcript. Session mechanics mirror DeepgramSession:
queue contract (frames / FINALIZE ignored / None -> CloseStream), same
reconnect-once with epochs and typed AsrUnavailable, same bounded shutdown,
same EventBuffer delivery (Updates replaceable, EndOfTurn never dropped).
"""
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

from server.realtime.asr import (CONNECT_TIMEOUT_SEC, FINALIZE,
                                 SHUTDOWN_TIMEOUT_SEC, AsrReconnected,
                                 AsrUnavailable)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FluxStartOfTurn:
    epoch: int = 1


@dataclass(frozen=True)
class FluxUpdate:
    transcript: str
    epoch: int = 1


@dataclass(frozen=True)
class FluxEndOfTurn:
    transcript: str
    epoch: int = 1


@dataclass(frozen=True)
class FluxEagerEndOfTurn:
    transcript: str
    epoch: int = 1


@dataclass(frozen=True)
class FluxTurnResumed:
    epoch: int = 1


FluxEvent = (FluxStartOfTurn | FluxUpdate | FluxEndOfTurn
             | FluxEagerEndOfTurn | FluxTurnResumed)


def build_flux_url(eot_threshold: float = 0.7,
                   eager_eot_threshold: float = 0.6) -> str:
    params = {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "eot_threshold": str(eot_threshold),
        "eager_eot_threshold": str(eager_eot_threshold),
    }
    return f"wss://api.deepgram.com/v2/listen?{urlencode(params)}"


def parse_flux_message(raw: str, epoch: int = 1) -> FluxEvent | None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if msg.get("type") != "TurnInfo":
        return None
    transcript = msg.get("transcript", "")
    match msg.get("event"):
        case "StartOfTurn":
            return FluxStartOfTurn(epoch=epoch)
        case "Update":
            return FluxUpdate(transcript=transcript, epoch=epoch)
        case "EagerEndOfTurn":
            return FluxEagerEndOfTurn(transcript=transcript, epoch=epoch)
        case "TurnResumed":
            return FluxTurnResumed(epoch=epoch)
        case "EndOfTurn":
            return FluxEndOfTurn(transcript=transcript, epoch=epoch)
        case _:
            return None


class FluxSession:
    def __init__(self, api_key: str, events_out: asyncio.Queue,
                 url: str | None = None) -> None:
        self._api_key = api_key
        self._events_out = events_out
        self._url = url or build_flux_url()
        self.dropped_events = 0

    async def run(self, audio: asyncio.Queue) -> None:
        for attempt in (1, 2):  # reconnect-once, matching DeepgramSession
            try:
                await self._run_once(audio, epoch=attempt)
                return
            except (websockets.ConnectionClosedError, OSError, TimeoutError) as exc:
                log.warning("flux connection lost (attempt %d): %s", attempt, exc)
                if attempt == 2:
                    raise AsrUnavailable("speech recognition unavailable") from exc
                self._events_out.put_nowait(AsrReconnected(epoch=attempt + 1))

    async def _run_once(self, audio: asyncio.Queue, epoch: int = 1) -> None:
        async with websockets.connect(
            self._url, additional_headers={"Authorization": f"Token {self._api_key}"},
            open_timeout=CONNECT_TIMEOUT_SEC,
        ) as ws:
            await self._pump(ws, audio, epoch=epoch)

    async def _pump(self, ws, audio: asyncio.Queue, epoch: int = 1) -> None:
        """Same bounded-shutdown contract as DeepgramSession._pump."""
        recv = asyncio.create_task(self._recv_loop(ws, epoch))
        try:
            await self._send_loop(ws, audio)
            try:
                await asyncio.wait_for(asyncio.shield(recv), SHUTDOWN_TIMEOUT_SEC)
            except TimeoutError:
                log.warning("flux shutdown flush timed out; forcing close")
        finally:
            if not recv.done():
                recv.cancel()
            with suppress(asyncio.CancelledError):
                await recv

    async def _send_loop(self, ws, audio: asyncio.Queue) -> None:
        while True:
            frame = await audio.get()
            if frame is None:
                await ws.send(json.dumps({"type": "CloseStream"}))
                return
            if frame is FINALIZE:
                continue  # v1-only concept; Flux owns its own endpointing
            await ws.send(frame)

    async def _recv_loop(self, ws, epoch: int = 1) -> None:
        async for raw in ws:
            event = parse_flux_message(raw, epoch=epoch)
            if event is None:
                continue
            self._events_out.put_nowait(event)
