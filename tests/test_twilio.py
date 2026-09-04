"""Twilio Media Streams adapter: the browser contract, spoken over the phone leg."""
import asyncio
import base64
import json

import numpy as np
import pytest

from server.realtime.protocol import encode_audio_frame
from server.realtime.transport import FRAME_BYTES, mulaw_decode
from server.realtime.twilio import PREBUFFER_SAMPLES, TwilioSocket, twiml_connect_stream


class FakeWs:
    """Stands in for Starlette's WebSocket: scripted inbound, recorded outbound."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.accepted = False
        self.closed: int | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        return await self.inbound.get()

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000) -> None:
        self.closed = code


def _start(sid: str = "MZ123") -> str:
    return json.dumps({"event": "start", "streamSid": sid,
                       "start": {"streamSid": sid, "callSid": "CA1",
                                 "mediaFormat": {"encoding": "audio/x-mulaw",
                                                 "sampleRate": 8000, "channels": 1}}})


def _media(mulaw: bytes) -> str:
    return json.dumps({"event": "media",
                       "media": {"payload": base64.b64encode(mulaw).decode()}})


def test_inbound_media_arrives_as_internal_frames() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        assert ws.accepted
        await ws.inbound.put(_start())
        await ws.inbound.put(_media(bytes([0xFF]) * 160))        # one 20 ms chunk
        await ws.inbound.put(_media(bytes([0xFF]) * 320))        # two chunks in one message
        frames = [await sock.receive() for _ in range(3)]
        assert all(len(f["bytes"]) == FRAME_BYTES for f in frames)
        assert sock.stream_sid == "MZ123"
        await sock.close()

    asyncio.run(run())


def test_partial_chunks_are_carried_never_padded() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(_start())
        await ws.inbound.put(_media(bytes([0xFF]) * 100))        # not yet a frame
        await ws.inbound.put(_media(bytes([0xFF]) * 60))         # completes it
        frame = await asyncio.wait_for(sock.receive(), timeout=0.5)
        assert len(frame["bytes"]) == FRAME_BYTES
        assert sock._inbound.empty()                              # exactly one frame
        await sock.close()

    asyncio.run(run())


def test_stop_and_socket_loss_both_read_as_disconnect() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(json.dumps({"event": "stop"}))
        assert (await sock.receive())["type"] == "websocket.disconnect"
        await sock.close()

    asyncio.run(run())


def test_outbound_audio_becomes_mulaw_media_addressed_to_the_stream() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(_start("MZ9"))
        await asyncio.sleep(0.01)
        t = np.arange(320) / 16000.0
        frame = (4000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()
        await sock.send_bytes(encode_audio_frame(7, frame))
        media = [m for m in ws.sent if m["event"] == "media"]
        assert media and media[0]["streamSid"] == "MZ9"
        mulaw = base64.b64decode(media[0]["media"]["payload"])
        assert len(mulaw) == 160
        assert np.abs(mulaw_decode(mulaw)).max() > 2000        # real signal, not silence
        await sock.close()

    asyncio.run(run())


def test_audio_end_becomes_a_mark_and_its_echo_is_the_drained_ack() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(_start())
        await asyncio.sleep(0.01)
        await sock.send_text(json.dumps({"type": "audio_end", "generation_id": 3}))
        marks = [m for m in ws.sent if m["event"] == "mark"]
        assert {"name": "gen-3"} in [m["mark"] for m in marks]
        await ws.inbound.put(json.dumps({"event": "mark", "mark": {"name": "gen-3"}}))
        ack = json.loads((await asyncio.wait_for(sock.receive(), timeout=0.5))["text"])
        assert ack == {"type": "playback_drained", "generation_id": 3}
        await sock.close()

    asyncio.run(run())


def test_clear_flushes_twilio_and_estimates_played_samples() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(_start())
        await asyncio.sleep(0.01)
        for _ in range(20):                                       # 20 frames sent
            await sock.send_bytes(encode_audio_frame(5, bytes(FRAME_BYTES)))
        await sock.send_text(json.dumps({"type": "clear", "generation_id": 5}))
        assert {"event": "clear", "streamSid": "MZ123"} in ws.sent
        ack = json.loads((await asyncio.wait_for(sock.receive(), timeout=0.5))["text"])
        assert ack["type"] == "cleared" and ack["generation_id"] == 5
        assert ack["played_samples"] == 20 * 320 - PREBUFFER_SAMPLES
        await sock.close()

    asyncio.run(run())


def test_the_call_id_is_handed_over_as_the_first_mark() -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "abc123")
        await sock.accept()
        await ws.inbound.put(_start())
        await asyncio.sleep(0.01)
        await sock.send_text(json.dumps({"type": "session", "call_id": "abc123"}))
        assert ws.sent[0] == {"event": "mark", "streamSid": "MZ123",
                              "mark": {"name": "call-abc123"}}
        await sock.close()

    asyncio.run(run())


@pytest.mark.parametrize("event", ["partial", "agent", "turn", "session_state", "vad"])
def test_ui_only_events_are_dropped_on_the_phone_leg(event: str) -> None:
    async def run() -> None:
        ws = FakeWs()
        sock = TwilioSocket(ws, "call-1")
        await sock.accept()
        await ws.inbound.put(_start())
        await asyncio.sleep(0.01)
        ws.sent.clear()
        await sock.send_text(json.dumps({"type": event, "text": "x"}))
        assert [m for m in ws.sent if m["event"] != "mark"] == []
        await sock.close()

    asyncio.run(run())


def test_twiml_connects_the_stream() -> None:
    xml = twiml_connect_stream("wss://example.ngrok.app/ws/twilio")
    assert xml.startswith('<?xml version="1.0"')
    assert '<Connect><Stream url="wss://example.ngrok.app/ws/twilio"/></Connect>' in xml
