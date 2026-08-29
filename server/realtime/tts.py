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

Two clients live here:
- TtsSession: one connection per utterance (the baseline). Measured cost: the
  ~800 ms connect handshake lands inside every reply's ttfb. Kept as the
  measured before/after reference.
- MultiContextTts: ONE connection per call (multi-stream-input), each utterance
  a "context" on it. A single reader task demuxes messages by contextId into
  per-context queues; synthesize() registers a queue, sends init/text/flush/
  close_context for its context, and yields audio until isFinal. Reconnects
  lazily if the socket dropped (inactivity), so worst case pays one handshake
  after a long silence instead of one per sentence. Cancelling a synthesize
  mid-stream best-effort closes its context so ElevenLabs stops generating.
  Each context queue is bounded; overflow discards that context's buffered audio
  and fails it closed instead of accumulating stale speech in memory.
"""
import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from urllib.parse import urlencode

import websockets

log = logging.getLogger(__name__)

FRAME_BYTES = 640  # 20 ms of 16 kHz PCM16
DEFAULT_MODEL = "eleven_flash_v2_5"
INACTIVITY_TIMEOUT_SEC = 180  # max allowed; the connection must outlive silences
TTS_CONTEXT_QUEUE_SIZE = 32   # provider chunks; overflow fails this context closed
TTS_CHUNK_TIMEOUT_SEC = 15    # max wait for the provider's next chunk
TTS_CONNECT_TIMEOUT_SEC = 10


class TtsBufferOverflow(RuntimeError):
    """The consumer could not keep up with provider audio for one context."""


class TtsTimeout(RuntimeError):
    """The provider stopped delivering audio for one context in time."""


def build_url(voice_id: str, model_id: str = DEFAULT_MODEL) -> str:
    params = {"model_id": model_id, "output_format": "pcm_16000"}
    return (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        f"?{urlencode(params)}"
    )


def build_multi_url(voice_id: str, model_id: str = DEFAULT_MODEL) -> str:
    params = {
        "model_id": model_id,
        "output_format": "pcm_16000",
        # We always send complete sentences, so skip the chunk-buffering
        # schedule entirely — the docs' recommended low-latency mode.
        "auto_mode": "true",
        "inactivity_timeout": str(INACTIVITY_TIMEOUT_SEC),
    }
    return (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input"
        f"?{urlencode(params)}"
    )


def parse_multi_message(raw: str) -> tuple[str, bytes, bool]:
    """Return (context_id, pcm_bytes, is_final) for one multi-context message."""
    msg = json.loads(raw)
    audio_b64 = msg.get("audio") or ""
    audio = base64.b64decode(audio_b64) if audio_b64 else b""
    return msg.get("contextId", ""), audio, bool(msg.get("isFinal"))


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


class MultiContextTts:
    def __init__(self, api_key: str, voice_id: str, url: str | None = None) -> None:
        self._api_key = api_key
        self._url = url or build_multi_url(voice_id)
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._counter = 0
        self._connect_lock = asyncio.Lock()

    async def ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and self._reader is not None and not self._reader.done():
                return
            self._ws = await websockets.connect(
                self._url, additional_headers={"xi-api-key": self._api_key},
                open_timeout=TTS_CONNECT_TIMEOUT_SEC,
            )
            self._reader = asyncio.create_task(self._read_loop(self._ws))

    async def _read_loop(self, ws) -> None:
        try:
            async for raw in ws:
                ctx, audio, is_final = parse_multi_message(raw)
                queue = self._queues.get(ctx)
                if queue is not None:
                    self._offer_context(queue, (audio, is_final))
        except websockets.ConnectionClosed as exc:
            log.info("tts connection closed: %s", exc)
        finally:
            for queue in list(self._queues.values()):
                self._fail_context(queue, ConnectionError("tts connection lost"))

    @staticmethod
    def _fail_context(queue: asyncio.Queue, error: Exception) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(error)

    def _offer_context(self, queue: asyncio.Queue, item) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            self._fail_context(
                queue, TtsBufferOverflow("tts context queue exceeded bounded capacity"))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM for one utterance. Cancel the generator to abort it."""
        await self.ensure_connected()
        assert self._ws is not None
        ctx = f"c{self._counter}"
        self._counter += 1
        queue: asyncio.Queue = asyncio.Queue(maxsize=TTS_CONTEXT_QUEUE_SIZE)
        self._queues[ctx] = queue
        finished = False
        try:
            await self._ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "context_id": ctx,
            }))
            await self._ws.send(json.dumps({"text": text + " ", "context_id": ctx}))
            await self._ws.send(json.dumps({"text": "", "flush": True, "context_id": ctx}))
            await self._ws.send(json.dumps({"context_id": ctx, "close_context": True}))
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(),
                                                  timeout=TTS_CHUNK_TIMEOUT_SEC)
                except TimeoutError as exc:
                    raise TtsTimeout(
                        f"no tts audio for context {ctx} within "
                        f"{TTS_CHUNK_TIMEOUT_SEC}s") from exc
                if isinstance(item, Exception):
                    raise item
                audio, is_final = item
                if audio:
                    yield audio
                if is_final:
                    finished = True
                    return
        finally:
            self._queues.pop(ctx, None)
            if not finished and self._ws is not None:  # barge-in: stop generating
                try:
                    await self._ws.send(json.dumps({"context_id": ctx,
                                                    "close_context": True}))
                except websockets.ConnectionClosed:
                    pass

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"close_socket": True}))
            except websockets.ConnectionClosed:
                pass
            await self._ws.close()
            self._ws = None


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
