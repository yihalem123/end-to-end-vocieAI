"""ElevenLabs streaming TTS client (raw WebSocket, no SDK). Phase 3.

## How this works
One wss connection per assistant reply to /v1/text-to-speech/{voice}/stream-input
(eleven_flash_v2_5, pcm_16000 — both free-tier). Protocol: an init message whose
text is a single space (it carries voice_settings), then text chunks each ending
in a space, then {"text": ""} as end-of-stream; audio returns as base64 PCM in
JSON messages until one arrives with isFinal=true. TtsSession.synthesize() is an
async generator yielding raw PCM as it arrives — the caller (call.py) owns
pacing, framing, and cancellation. Cancelling the generator mid-reply (barge-in)
just abandons the connection; the context manager closes the socket, ElevenLabs
stops billing at the audio it generated.

FrameChunker reslices ElevenLabs' arbitrary-size chunks into our 640-byte wire
frames, carrying remainders across pushes; flush() pads the final partial frame
with silence so the playback worklet never sees a short frame.

Latency note (measured in Phase 3 profiling): the per-reply connect handshake is
part of tts_ttfb here. The production fix is a pre-warmed or multi-context
connection — a known refinement, deliberately not built until measurement says so.
"""
import base64
import json
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets

FRAME_BYTES = 640  # 20 ms of 16 kHz PCM16
DEFAULT_MODEL = "eleven_flash_v2_5"


def build_url(voice_id: str, model_id: str = DEFAULT_MODEL) -> str:
    params = {"model_id": model_id, "output_format": "pcm_16000"}
    return (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        f"?{urlencode(params)}"
    )


def parse_tts_message(raw: str) -> tuple[bytes, bool]:
    """Return (pcm_bytes, is_final) for one ElevenLabs message."""
    msg = json.loads(raw)
    audio_b64 = msg.get("audio") or ""
    audio = base64.b64decode(audio_b64) if audio_b64 else b""
    return audio, bool(msg.get("isFinal"))


class FrameChunker:
    def __init__(self) -> None:
        self._pending = b""

    def push(self, chunk: bytes):
        data = self._pending + chunk
        for off in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
            yield data[off : off + FRAME_BYTES]
        self._pending = data[len(data) - len(data) % FRAME_BYTES :]

    def flush(self) -> bytes | None:
        if not self._pending:
            return None
        frame = self._pending + bytes(FRAME_BYTES - len(self._pending))
        self._pending = b""
        return frame


class TtsSession:
    def __init__(self, api_key: str, voice_id: str, url: str | None = None) -> None:
        self._api_key = api_key
        self._url = url or build_url(voice_id)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM for one reply. Cancel the generator to abort mid-reply."""
        async with websockets.connect(
            self._url, additional_headers={"xi-api-key": self._api_key}
        ) as ws:
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                # Generate on smaller buffers first: lower time-to-first-byte.
                "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
            }))
            await ws.send(json.dumps({"text": text + " "}))
            await ws.send(json.dumps({"text": ""}))  # end-of-stream: flush + close
            async for raw in ws:
                audio, is_final = parse_tts_message(raw)
                if audio:
                    yield audio
                if is_final:
                    return
