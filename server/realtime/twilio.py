"""Twilio Media Streams as a transport: the phone leg without touching the engine.

## How this works
Twilio connects a WebSocket to /ws/twilio and speaks JSON: `start` (with the
streamSid we must echo), `media` (base64 mu-law @ 8 kHz, 160 bytes per 20 ms),
`mark` (echoed back once the audio queued before it has played), `stop`.
TwilioSocket wraps that socket so CallSession sees exactly the browser
contract and nothing else changes:
- inbound `media` becomes 640-byte internal frames (transport.py codecs) that
  receive() hands over as {"bytes": ...}; `stop` becomes a disconnect
- outbound audio: send_bytes() takes the generation-prefixed wire frame the
  Speaker already produces, transcodes to mu-law and emits a `media` event
- the browser's playback protocol maps onto Twilio's: an `audio_end` event
  becomes a `mark` named after the generation, and Twilio echoing that mark IS
  the `playback_drained` ack; a `clear` event becomes Twilio's `clear`, and the
  `cleared` ack carries played samples ESTIMATED from what was sent minus the
  Speaker's prebuffer (Twilio does not report playback position)
- every other JSON event (partials, transcript cards, metrics) has no phone
  UI to go to and is dropped.
A reader task translates Twilio messages into a queue so synthesized acks can
be interleaved with real ones; accept() starts it and close() cancels it.
Marks are also used to hand the call id to a client on connect, which is how
the offline Twilio-protocol simulator finds its report.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
from collections import deque
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect

from server.realtime.protocol import decode_audio_frame
from server.realtime.transport import (
    FRAME_SAMPLES,
    TELEPHONY_FRAME_BYTES,
    frame_to_twilio_payload,
    twilio_media_to_frame,
)

log = logging.getLogger(__name__)

# The Speaker sends a PREBUFFER_FRAMES burst before pacing, so at any moment
# roughly that much sent audio is still unplayed. Used only to estimate the
# `cleared` ack's played_samples; the browser leg reports the real number.
PREBUFFER_SAMPLES = 10 * FRAME_SAMPLES


class TwilioSocket:
    """Duck-types the subset of the WebSocket API CallSession uses."""

    def __init__(self, ws: WebSocket, call_id: str) -> None:
        self._ws = ws
        self._call_id = call_id
        self.stream_sid: str | None = None
        self._inbound: asyncio.Queue = asyncio.Queue()
        self._frames: deque[bytes] = deque()
        self._carry = b""
        self._reader: asyncio.Task | None = None
        self._sent_samples: dict[int, int] = {}
        self._pending_marks: dict[str, int] = {}

    # --- lifecycle ----------------------------------------------------------

    async def accept(self) -> None:
        await self._ws.accept()
        self._reader = asyncio.create_task(self._read_loop())

    async def close(self, code: int = 1000) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        with suppress(RuntimeError, WebSocketDisconnect):
            await self._ws.close(code=code)

    # --- inbound: Twilio JSON -> CallSession messages -----------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    raw = await self._ws.receive_text()
                except WebSocketDisconnect:
                    await self._inbound.put({"type": "websocket.disconnect"})
                    return
                for message in self.translate_inbound(raw):
                    await self._inbound.put(message)
                    if message.get("type") == "websocket.disconnect":
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("twilio reader failed; ending call %s", self._call_id)
            await self._inbound.put({"type": "websocket.disconnect"})

    def translate_inbound(self, raw: str) -> list[dict]:
        """One Twilio message -> zero or more CallSession-shaped messages."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out: list[dict] = []
        match msg.get("event"):
            case "start":
                self.stream_sid = (msg.get("start") or {}).get("streamSid") or msg.get("streamSid")
                # Hand the call id to the far side as a mark; Twilio echoes a
                # mark immediately when nothing is queued, and the simulator
                # uses it to find its report. Harmless on a real call.
                self._outbound_marks_on_start = True
            case "media":
                payload = (msg.get("media") or {}).get("payload") or ""
                self._carry += base64.b64decode(payload)
                while len(self._carry) >= TELEPHONY_FRAME_BYTES:
                    chunk, self._carry = (self._carry[:TELEPHONY_FRAME_BYTES],
                                          self._carry[TELEPHONY_FRAME_BYTES:])
                    out.append({"type": "websocket.receive",
                                "bytes": twilio_media_to_frame(chunk)})
            case "mark":
                name = (msg.get("mark") or {}).get("name") or ""
                generation = self._pending_marks.pop(name, None)
                if generation is not None:
                    out.append({"type": "websocket.receive", "text": json.dumps(
                        {"type": "playback_drained", "generation_id": generation})})
            case "stop":
                out.append({"type": "websocket.disconnect"})
        return out

    async def receive(self) -> dict:
        return await self._inbound.get()

    # --- outbound: CallSession messages -> Twilio JSON ----------------------

    async def send_text(self, data: str) -> None:
        """Server JSON events: only the playback protocol has a phone-side meaning."""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return
        if getattr(self, "_outbound_marks_on_start", False) and self.stream_sid:
            self._outbound_marks_on_start = False
            await self._emit({"event": "mark", "streamSid": self.stream_sid,
                              "mark": {"name": f"call-{self._call_id}"}})
        match event.get("type"):
            case "audio_end":
                generation = int(event.get("generation_id", 0))
                name = f"gen-{generation}"
                self._pending_marks[name] = generation
                await self._emit({"event": "mark", "streamSid": self.stream_sid,
                                  "mark": {"name": name}})
            case "clear":
                generation = int(event.get("generation_id", 0))
                await self._emit({"event": "clear", "streamSid": self.stream_sid})
                played = max(0, self._sent_samples.get(generation, 0) - PREBUFFER_SAMPLES)
                await self._inbound.put({"type": "websocket.receive", "text": json.dumps(
                    {"type": "cleared", "generation_id": generation,
                     "played_samples": played})})
            case _:
                pass  # no phone-side UI for partials, cards, metrics, state

    async def send_bytes(self, wire: bytes) -> None:
        generation, frame = decode_audio_frame(wire)
        self._sent_samples[generation] = self._sent_samples.get(generation, 0) + FRAME_SAMPLES
        await self._emit({"event": "media", "streamSid": self.stream_sid,
                          "media": {"payload": frame_to_twilio_payload(frame)}})

    async def _emit(self, message: dict) -> None:
        if self.stream_sid is None and message.get("event") != "clear":
            return  # nothing to address until Twilio has said `start`
        await self._ws.send_text(json.dumps(message))


def twilio_signature(url: str, params: dict[str, str], auth_token: str) -> str:
    """Twilio's request signature: HMAC-SHA1 over the URL plus the POST params
    concatenated key+value in sorted key order, base64-encoded."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(digest.digest()).decode("ascii")


def verify_twilio_signature(url: str, params: dict[str, str], auth_token: str,
                            signature: str | None) -> bool:
    if not signature:
        return False
    return hmac.compare_digest(twilio_signature(url, params, auth_token), signature)


def twiml_connect_stream(ws_url: str) -> str:
    """TwiML that connects the call's audio to our media-stream socket."""
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Connect><Stream url="{ws_url}"/></Connect></Response>')
