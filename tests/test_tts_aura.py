"""Deepgram Aura speak-streaming adapter: URL, protocol, failure modes."""
import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from server.realtime.tts import TtsNoAudio, TtsTimeout
from server.realtime.tts_aura import AuraTts, build_aura_url


def test_build_aura_url_pins_model_and_audio() -> None:
    parsed = urlparse(build_aura_url("aura-2-thalia-en"))
    assert parsed.path == "/v1/speak"
    q = parse_qs(parsed.query)
    assert q["model"] == ["aura-2-thalia-en"]
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]


class ScriptedWs:
    """Duck-typed websocket: records sends, plays back a scripted recv queue."""

    def __init__(self, script) -> None:
        self.sent: list[dict] = []
        self._script = asyncio.Queue()
        for item in script:
            self._script.put_nowait(item)

    async def send(self, data) -> None:
        self.sent.append(json.loads(data))

    async def recv(self):
        return await self._script.get()

    def feed(self, item) -> None:
        self._script.put_nowait(item)

    async def close(self) -> None:
        pass


def _client(ws) -> AuraTts:
    client = object.__new__(AuraTts)
    client._api_key = "k"
    client._url = "wss://unused"
    client._ws = ws
    client._lock = asyncio.Lock()
    client._connect_lock = asyncio.Lock()
    client._reclaim_task = None
    return client


def test_synthesize_speaks_flushes_and_yields_until_flushed() -> None:
    ws = ScriptedWs([b"aa", b"bb", json.dumps({"type": "Flushed"})])
    client = _client(ws)

    async def run() -> None:
        chunks = [c async for c in client.synthesize("Hello there.")]
        assert chunks == [b"aa", b"bb"]
        assert [m["type"] for m in ws.sent] == ["Speak", "Flush"]
        assert ws.sent[0]["text"] == "Hello there."
        assert not client._lock.locked()  # released for the next utterance

    asyncio.run(run())


def test_flushed_with_zero_audio_raises_tts_no_audio() -> None:
    # Same failure contract as the ElevenLabs client: an exhausted quota or
    # provider rejection must never ship a silent success.
    ws = ScriptedWs([json.dumps({"type": "Metadata"}),
                     json.dumps({"type": "Flushed"})])
    client = _client(ws)

    async def run() -> None:
        with pytest.raises(TtsNoAudio):
            async for _ in client.synthesize("Hello."):
                raise AssertionError("no audio should arrive")

    asyncio.run(run())


def test_provider_stall_raises_typed_timeout(monkeypatch) -> None:
    from server.realtime import tts_aura
    monkeypatch.setattr(tts_aura, "TTS_CHUNK_TIMEOUT_SEC", 0.01)
    ws = ScriptedWs([])  # never answers
    client = _client(ws)

    async def run() -> None:
        with pytest.raises(TtsTimeout):
            async for _ in client.synthesize("Hello."):
                pass

    asyncio.run(run())


def test_provider_trickle_cannot_exceed_receive_wait_budget(monkeypatch) -> None:
    from server.realtime import tts_aura
    monkeypatch.setattr(tts_aura, "AURA_RECEIVE_WAIT_BUDGET_SEC", 0.03)
    monkeypatch.setattr(tts_aura, "TTS_CHUNK_TIMEOUT_SEC", 1.0)

    class TricklingWs(ScriptedWs):
        def __init__(self) -> None:
            super().__init__([])

        async def recv(self):
            await asyncio.sleep(0.01)
            return b"still-going"

    client = _client(TricklingWs())

    async def run() -> None:
        with pytest.raises(TtsTimeout, match="receive-wait budget"):
            async for _ in client.synthesize("A provider that never flushes."):
                pass

    asyncio.run(run())


def test_slow_consumer_pacing_does_not_spend_provider_wait_budget(
    monkeypatch,
) -> None:
    """Speaker deliberately paces PCM after each yield; that is not a stall."""
    from server.realtime import tts_aura
    monkeypatch.setattr(tts_aura, "AURA_RECEIVE_WAIT_BUDGET_SEC", 0.02)
    monkeypatch.setattr(tts_aura, "TTS_CHUNK_TIMEOUT_SEC", 1.0)
    ws = ScriptedWs([b"one", b"two", json.dumps({"type": "Flushed"})])
    client = _client(ws)

    async def run() -> None:
        chunks = []
        async for chunk in client.synthesize("A healthy long utterance."):
            chunks.append(chunk)
            await asyncio.sleep(0.03)  # longer than the entire provider budget
        assert chunks == [b"one", b"two"]

    asyncio.run(run())


def test_aborted_synthesis_clears_and_releases_after_cleared() -> None:
    # Barge-in: the generator is closed mid-stream. The adapter must send
    # Clear, swallow the utterance's residual audio, and only then free the
    # single stream for the next generation's synthesize().
    ws = ScriptedWs([b"aa", b"stale-1", b"stale-2"])
    client = _client(ws)

    async def run() -> None:
        gen = client.synthesize("A long reply that gets interrupted.")
        assert await anext(gen) == b"aa"
        await gen.aclose()  # barge-in cancels the producer
        await asyncio.sleep(0.01)  # let the detached reclaim task run
        assert {"type": "Clear"} in ws.sent
        assert client._lock.locked()  # held until the stream is clean
        ws.feed(json.dumps({"type": "Cleared"}))
        await asyncio.sleep(0.05)
        assert not client._lock.locked()

    asyncio.run(run())


def test_cancellation_while_awaiting_audio_also_reclaims_the_stream() -> None:
    # Barge-in usually cancels the producer while it's BLOCKED in recv(), not
    # parked at a yield — the reclaim path must cover both suspension points.
    ws = ScriptedWs([b"aa"])  # provider mid-utterance, next chunk never comes
    client = _client(ws)

    async def consume() -> None:
        async for _ in client.synthesize("A long interrupted reply."):
            pass

    async def run() -> None:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # chunk consumed; now blocked in recv
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.01)
        assert {"type": "Clear"} in ws.sent
        assert client._lock.locked()  # held until the stream is clean
        ws.feed(json.dumps({"type": "Cleared"}))
        await asyncio.sleep(0.05)
        assert not client._lock.locked()

    asyncio.run(run())


def test_only_protocol_messages_are_ever_sent() -> None:
    # Live regression: a KeepAlive ping (an STT-socket habit) is NOT part of
    # the speak protocol — Deepgram closes the stream 1008 DATA-0000 on it.
    ws = ScriptedWs([b"one", json.dumps({"type": "Flushed"})])
    client = _client(ws)

    async def run() -> None:
        [c async for c in client.synthesize("Hello.")]
        await asyncio.sleep(0.02)
        assert {m["type"] for m in ws.sent} <= {"Speak", "Flush", "Clear", "Close"}

    asyncio.run(run())


def test_second_utterance_waits_for_the_single_stream() -> None:
    ws = ScriptedWs([b"one", json.dumps({"type": "Flushed"}),
                     b"two", json.dumps({"type": "Flushed"})])
    client = _client(ws)

    async def run() -> None:
        first = [c async for c in client.synthesize("First.")]
        second = [c async for c in client.synthesize("Second.")]
        assert (first, second) == ([b"one"], [b"two"])
        speaks = [m["text"] for m in ws.sent if m["type"] == "Speak"]
        assert speaks == ["First.", "Second."]

    asyncio.run(run())


def test_cancellation_during_connect_does_not_crash_the_reclaim() -> None:
    # Found live: a barge-in landed while the very first Speak was still
    # opening the socket, so there was no connection to scrub. The reclaim task
    # died on None.send() and left a stack trace in the logs each call.
    from server.realtime import tts_aura

    async def run() -> None:
        client = _client(None)

        connecting = asyncio.Event()

        async def slow_connect() -> None:
            connecting.set()
            await asyncio.sleep(3600)  # never completes; cancelled mid-connect

        client.ensure_connected = slow_connect

        async def consume() -> None:
            async for _ in client.synthesize("Hello."):
                pass

        task = asyncio.create_task(consume())
        await connecting.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.05)
        assert not client._lock.locked()  # freed for the next utterance
        # Nothing was ever streamed, so there is nothing to scrub: scheduling a
        # reclaim here is what crashed on None.send().
        assert client._reclaim_task is None

    asyncio.run(run())


# --- warm socket cache: keep the ~2.4 s connect off the first utterance ---

class FakeSocket:
    def __init__(self) -> None:
        self.close_code = None
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        self.close_code = 1000


def _cache(**kw):
    from server.realtime.tts_aura import WarmSocketCache
    made: list[FakeSocket] = []

    async def connect(_url, _key):
        made.append(FakeSocket())
        return made[-1]

    return WarmSocketCache(connect=connect, **kw), made


def test_warm_socket_is_handed_over_once() -> None:
    async def run() -> None:
        cache, made = _cache()
        cache.prewarm("wss://a", "key")
        await asyncio.sleep(0)
        assert await cache.take("wss://a") is made[0]
        assert await cache.take("wss://a") is None   # handed over, not shared
        await cache.close()

    asyncio.run(run())


def test_stale_warm_socket_is_closed_rather_than_handed_over() -> None:
    # Measured: an idle speak stream's first byte degrades (364 ms fresh vs
    # 614 ms after 25 s), so an old socket is worse than a fresh connect.
    async def run() -> None:
        cache, made = _cache(max_age_sec=0.0)
        cache.prewarm("wss://a", "key")
        await asyncio.sleep(0)
        assert await cache.take("wss://a") is None
        assert made[0].closed
        await cache.close()

    asyncio.run(run())


def test_dead_or_mismatched_warm_socket_is_never_handed_over() -> None:
    async def run() -> None:
        cache, made = _cache()
        cache.prewarm("wss://a", "key")
        await asyncio.sleep(0)
        made[0].close_code = 1006                    # provider dropped it
        assert await cache.take("wss://a") is None

        cache.prewarm("wss://a", "key")
        await asyncio.sleep(0)
        assert await cache.take("wss://different-voice") is None
        await cache.close()

    asyncio.run(run())


def test_prewarm_does_not_stack_connections() -> None:
    async def run() -> None:
        cache, made = _cache()
        for _ in range(4):
            cache.prewarm("wss://a", "key")
        await asyncio.sleep(0.01)
        assert len(made) == 1                        # one in flight, one kept
        await cache.close()
        assert made[0].closed                        # shutdown releases it

    asyncio.run(run())


def test_ensure_connected_prefers_the_warm_socket() -> None:
    from server.realtime import tts_aura

    async def run() -> None:
        cache, made = _cache()
        cache.prewarm("wss://unused", "k")
        await asyncio.sleep(0)

        client = object.__new__(tts_aura.AuraTts)
        client._api_key = "k"
        client._url = "wss://unused"
        client._ws = None
        client._connect_lock = asyncio.Lock()
        client._warm = cache

        async def explode(*_a, **_kw):  # a real connect would cost ~2.4 s
            raise AssertionError("should have used the warm socket")

        tts_aura.websockets.connect = explode
        try:
            await client.ensure_connected()
            assert client._ws is made[0]
        finally:
            tts_aura.websockets = __import__("websockets")

    asyncio.run(run())
