"""Deepgram Aura TTS over the speak-streaming WebSocket.

## How this works
One persistent connection per call, one utterance in flight at a time — which
matches the Speaker exactly: its producer consumes sentences sequentially, so
a single stream loses nothing over ElevenLabs' multi-context fan-out. Each
synthesize() takes the stream lock, sends Speak + Flush, and yields binary
audio until the provider's Flushed marker. Barge-in aborts the generator
mid-stream (at recv or at a yield): a detached reclaim task sends Clear,
swallows the aborted utterance's residual audio until Cleared, and only then
releases the lock, so the next generation never hears stale frames. The
protocol accepts ONLY Speak/Flush/Clear/Close — a KeepAlive ping (an STT
habit) gets the stream closed 1008 DATA-0000 (found live); idle drops are
handled instead by reconnecting once on the next Speak. Zero-audio Flushed
raises TtsNoAudio (an exhausted quota must fail loud, not ship a silent call).
"""
import asyncio
import json
import logging
from contextlib import suppress
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

from server.realtime.tts import (
    TTS_CHUNK_TIMEOUT_SEC,
    TTS_CONNECT_TIMEOUT_SEC,
    TtsNoAudio,
    TtsTimeout,
)

log = logging.getLogger(__name__)

RECLAIM_TIMEOUT_SEC = 3  # max wait for Cleared while flushing stale audio
AURA_UTTERANCE_TIMEOUT_SEC = 15  # total cap; trickling chunks cannot run forever


def build_aura_url(model: str) -> str:
    params = {"model": model, "encoding": "linear16", "sample_rate": "16000"}
    return f"wss://api.deepgram.com/v1/speak?{urlencode(params)}"


class AuraTts:
    def __init__(self, api_key: str, model: str = "aura-2-thalia-en",
                 url: str | None = None) -> None:
        self._api_key = api_key
        self._url = url or build_aura_url(model)
        self._ws = None
        self._lock = asyncio.Lock()          # one utterance owns the stream
        self._connect_lock = asyncio.Lock()
        # Kept so a detached scrub cannot be garbage-collected mid-flight and
        # so its failures surface instead of vanishing into the GC.
        self._reclaim_task: asyncio.Task | None = None

    async def ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._ws is not None:
                return
            self._ws = await websockets.connect(
                self._url, additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=TTS_CONNECT_TIMEOUT_SEC,
            )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM for one utterance. Cancel the generator to abort it."""
        await self._lock.acquire()
        release_here = True
        try:
            try:
                try:
                    async with asyncio.timeout(AURA_UTTERANCE_TIMEOUT_SEC):
                        await self._send_utterance(text)
                        yielded_any = False
                        while True:
                            try:
                                msg = await asyncio.wait_for(
                                    self._ws.recv(), timeout=TTS_CHUNK_TIMEOUT_SEC)
                            except TimeoutError as exc:
                                await self._drop_connection()  # stream state unknown
                                raise TtsTimeout(
                                    f"no aura audio within {TTS_CHUNK_TIMEOUT_SEC}s"
                                ) from exc
                            if isinstance(msg, bytes):
                                yielded_any = True
                                yield msg
                                continue
                            event = json.loads(msg)
                            if event.get("type") == "Flushed":
                                if not yielded_any:
                                    raise TtsNoAudio(
                                        "aura flushed with zero audio "
                                        "(quota exhausted or provider rejection)")
                                return
                            if event.get("type") == "Warning":
                                log.warning("aura warning: %s",
                                            event.get("description"))
                except TimeoutError as exc:
                    await self._drop_connection()
                    raise TtsTimeout(
                        "aura utterance exceeded total timeout of "
                        f"{AURA_UTTERANCE_TIMEOUT_SEC}s"
                    ) from exc
            except (asyncio.CancelledError, GeneratorExit):
                # Aborted mid-utterance (barge-in): hand the lock to a reclaim
                # task that scrubs the stream, then let the abort propagate.
                # An abort during connect has no stream to scrub — and no
                # connection to hand over (found live: a barge-in landed while
                # the first Speak was still opening the socket).
                if self._ws is None:
                    raise
                release_here = False
                self._reclaim_task = asyncio.create_task(self._reclaim(self._ws))
                raise
        finally:
            if release_here:
                self._lock.release()

    async def _send_utterance(self, text: str) -> None:
        await self.ensure_connected()
        try:
            await self._ws.send(json.dumps({"type": "Speak", "text": text}))
        except websockets.ConnectionClosed:
            self._ws = None  # idle drop between turns: reconnect once
            await self.ensure_connected()
            await self._ws.send(json.dumps({"type": "Speak", "text": text}))
        await self._ws.send(json.dumps({"type": "Flush"}))

    async def _reclaim(self, ws) -> None:
        """After an aborted utterance: Clear, then discard until Cleared."""
        try:
            async with asyncio.timeout(RECLAIM_TIMEOUT_SEC):
                await ws.send(json.dumps({"type": "Clear"}))
                while True:
                    msg = await ws.recv()
                    if not isinstance(msg, bytes) and (
                            json.loads(msg).get("type") == "Cleared"):
                        return
        except (websockets.ConnectionClosed, TimeoutError, OSError):
            log.warning("aura reclaim failed; dropping connection")
            if self._ws is ws:
                await self._drop_connection()
        finally:
            self._lock.release()

    async def _drop_connection(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except OSError:  # already torn down
                pass

    async def close(self) -> None:
        if self._reclaim_task is not None:
            self._reclaim_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reclaim_task
            self._reclaim_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "Close"}))
                await ws.close()
            except (websockets.ConnectionClosed, OSError):
                pass
