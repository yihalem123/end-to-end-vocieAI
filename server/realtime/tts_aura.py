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
import time
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
# Opening a speak socket measured a median 2377 ms (n=6, 1.3-2.7 s), and it is
# paid on a call's FIRST utterance - the greeting, the worst place to spend it.
# Warming one before the call exists moves that cost off the call entirely.
# Warm sockets go stale though: first byte measured 364 ms fresh vs 614 ms
# after 25 s idle, so past this age a fresh connect is the better trade.
WARM_MAX_AGE_SEC = 180
AURA_RECEIVE_WAIT_BUDGET_SEC = 15  # cumulative provider wait; excludes pacing


def build_aura_url(model: str) -> str:
    params = {"model": model, "encoding": "linear16", "sample_rate": "16000"}
    return f"wss://api.deepgram.com/v1/speak?{urlencode(params)}"


async def _open_socket(url: str, api_key: str):
    return await websockets.connect(
        url, additional_headers={"Authorization": f"Token {api_key}"},
        open_timeout=TTS_CONNECT_TIMEOUT_SEC,
    )


async def _close_quietly(ws) -> None:
    try:
        await ws.close()
    except (websockets.ConnectionClosed, OSError):
        pass


class WarmSocketCache:
    """At most one pre-opened speak socket, handed to the next call that needs
    it. Never shared: take() gives the socket away, so two calls can never end
    up driving the same single-utterance stream."""

    def __init__(self, connect=None, max_age_sec: float = WARM_MAX_AGE_SEC) -> None:
        self._connect = connect or _open_socket
        self._max_age = max_age_sec
        self._ws = None
        self._url = ""
        self._opened_t = 0.0
        self._task: asyncio.Task | None = None

    def prewarm(self, url: str, api_key: str) -> None:
        """Fire-and-forget. Safe to call on every page load: one connection is
        kept, and a fill already in flight is never duplicated."""
        if self._task is not None and not self._task.done():
            return
        if self._usable(url):
            return
        self._task = asyncio.create_task(self._fill(url, api_key))

    async def _fill(self, url: str, api_key: str) -> None:
        try:
            ws = await self._connect(url, api_key)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("aura prewarm failed; calls will connect on demand",
                        exc_info=True)
            return
        old, self._ws = self._ws, ws
        self._url, self._opened_t = url, time.monotonic()
        if old is not None:
            await _close_quietly(old)

    def _usable(self, url: str) -> bool:
        return (self._ws is not None
                and self._url == url
                and getattr(self._ws, "close_code", None) is None
                and time.monotonic() - self._opened_t < self._max_age)

    async def take(self, url: str):
        """Hand over the warm socket, or None if there is nothing fresh."""
        if self._ws is None:
            return None
        ws, usable = self._ws, self._usable(url)
        self._ws, self._url = None, ""
        if usable:
            return ws
        await _close_quietly(ws)
        return None

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._ws is not None:
            ws, self._ws = self._ws, None
            await _close_quietly(ws)


warm_sockets = WarmSocketCache()  # process-wide; app lifespan owns its shutdown


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
        self._warm = warm_sockets

    async def ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._ws is not None:
                return
            self._ws = (await self._warm.take(self._url)
                        or await _open_socket(self._url, self._api_key))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM for one utterance. Cancel the generator to abort it."""
        await self._lock.acquire()
        release_here = True
        try:
            try:
                await self._send_utterance(text)
                yielded_any = False
                receive_wait_sec = 0.0
                while True:
                    remaining = AURA_RECEIVE_WAIT_BUDGET_SEC - receive_wait_sec
                    if remaining <= 0:
                        await self._drop_connection()
                        raise TtsTimeout(
                            "aura exceeded cumulative receive-wait budget of "
                            f"{AURA_RECEIVE_WAIT_BUDGET_SEC}s")
                    budget_limited = remaining <= TTS_CHUNK_TIMEOUT_SEC
                    started = time.monotonic()
                    try:
                        msg = await asyncio.wait_for(
                            self._ws.recv(),
                            timeout=min(TTS_CHUNK_TIMEOUT_SEC, remaining),
                        )
                    except TimeoutError as exc:
                        await self._drop_connection()  # stream state unknown
                        if budget_limited:
                            raise TtsTimeout(
                                "aura exceeded cumulative receive-wait budget of "
                                f"{AURA_RECEIVE_WAIT_BUDGET_SEC}s"
                            ) from exc
                        raise TtsTimeout(
                            f"no aura audio within {TTS_CHUNK_TIMEOUT_SEC}s"
                        ) from exc
                    receive_wait_sec += time.monotonic() - started
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
