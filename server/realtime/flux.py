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
TurnResumed/EagerEndOfTurn belong to eager mode (speculative LLM start), the
documented stretch. Session mechanics mirror asr.DeepgramSession: same audio
queue contract (frames / FINALIZE ignored / None -> CloseStream), same
reconnect-once, same bounded drop-newest event delivery.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

from server.realtime.asr import FINALIZE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FluxStartOfTurn:
    pass


@dataclass(frozen=True)
class FluxUpdate:
    transcript: str


@dataclass(frozen=True)
class FluxEndOfTurn:
    transcript: str


FluxEvent = FluxStartOfTurn | FluxUpdate | FluxEndOfTurn


def build_flux_url(eot_threshold: float = 0.7) -> str:
    params = {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "eot_threshold": str(eot_threshold),
    }
    return f"wss://api.deepgram.com/v2/listen?{urlencode(params)}"


def parse_flux_message(raw: str) -> FluxEvent | None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if msg.get("type") != "TurnInfo":
        return None
    transcript = msg.get("transcript", "")
    match msg.get("event"):
        case "StartOfTurn":
            return FluxStartOfTurn()
        case "Update":
            return FluxUpdate(transcript=transcript)
        case "EndOfTurn":
            return FluxEndOfTurn(transcript=transcript)
        case _:  # TurnResumed / EagerEndOfTurn: eager mode only (stretch)
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
                await self._run_once(audio)
                return
            except (websockets.ConnectionClosedError, OSError) as exc:
                log.warning("flux connection lost (attempt %d): %s", attempt, exc)
                if attempt == 2:
                    raise

    async def _run_once(self, audio: asyncio.Queue) -> None:
        async with websockets.connect(
            self._url, additional_headers={"Authorization": f"Token {self._api_key}"}
        ) as ws:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._send_loop(ws, audio))
                tg.create_task(self._recv_loop(ws))

    async def _send_loop(self, ws, audio: asyncio.Queue) -> None:
        while True:
            frame = await audio.get()
            if frame is None:
                await ws.send(json.dumps({"type": "CloseStream"}))
                return
            if frame is FINALIZE:
                continue  # v1-only concept; Flux owns its own endpointing
            await ws.send(frame)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            event = parse_flux_message(raw)
            if event is None:
                continue
            try:
                self._events_out.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_events += 1
