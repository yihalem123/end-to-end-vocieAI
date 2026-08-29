"""Deepgram URL construction, message parsing, and send-loop control frames."""
import asyncio
import json
from urllib.parse import parse_qs, urlparse

from server.realtime.asr import (
    FINALIZE,
    AsrFinal,
    AsrPartial,
    AsrReconnected,
    AsrUtteranceEnd,
    DeepgramSession,
    build_url,
    parse_message,
)


def test_build_url_pins_the_planned_params() -> None:
    url = build_url()
    parsed = urlparse(url)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.deepgram.com"
    assert parsed.path == "/v1/listen"
    q = parse_qs(parsed.query)
    assert q["model"] == ["nova-3"]
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]
    assert q["channels"] == ["1"]
    assert q["interim_results"] == ["true"]   # required by utterance_end_ms
    assert q["endpointing"] == ["300"]
    assert q["utterance_end_ms"] == ["1000"]
    assert q["punctuate"] == ["true"]         # endpointer's looks_complete needs it
    assert q["filler_words"] == ["true"]      # "um" is trailing-word evidence


def _results(transcript: str, is_final: bool, speech_final: bool = False) -> str:
    return json.dumps({
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript}]},
    })


def test_parse_partial() -> None:
    event = parse_message(_results("i have five", is_final=False))
    assert event == AsrPartial(text="i have five")


def test_parse_final_with_speech_final() -> None:
    event = parse_message(_results("I have five years.", is_final=True, speech_final=True))
    assert event == AsrFinal(text="I have five years.", speech_final=True)


def test_parse_utterance_end() -> None:
    event = parse_message(json.dumps({"type": "UtteranceEnd", "last_word_end": 3.1}))
    assert event == AsrUtteranceEnd()


def test_parse_ignores_metadata_and_empty_partials() -> None:
    assert parse_message(json.dumps({"type": "Metadata", "request_id": "x"})) is None
    assert parse_message(_results("", is_final=False)) is None  # noise between words
    # empty FINALS still parse: endpointer treats them as no-ops, but silently
    # dropping them here would hide Deepgram behavior from logs
    assert parse_message(_results("", is_final=True)) == AsrFinal(text="", speech_final=False)


def test_parse_garbage_returns_none() -> None:
    assert parse_message("not json at all") is None
    assert parse_message(json.dumps({"no": "type"})) is None


class FakeWs:
    def __init__(self, hang_recv: bool = False) -> None:
        self.sent: list = []
        self._hang_recv = hang_recv

    async def send(self, data) -> None:
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._hang_recv:
            await asyncio.Event().wait()  # a server that never closes
        raise StopAsyncIteration


def test_send_loop_control_frames() -> None:
    # Latency fix (2026-08-31): short utterances ("Five.") took 1.5-3.5s to
    # finalize on Deepgram's own endpointing. Our VAD knows when speech stopped,
    # so a FINALIZE marker in the audio queue becomes {"type":"Finalize"},
    # forcing finals to flush immediately. None still becomes CloseStream.
    async def run() -> list:
        ws = FakeWs()
        session = DeepgramSession("key", asyncio.Queue())
        audio: asyncio.Queue = asyncio.Queue()
        audio.put_nowait(b"\x00" * 640)
        audio.put_nowait(FINALIZE)
        audio.put_nowait(None)
        await session._send_loop(ws, audio)
        return ws.sent

    sent = asyncio.run(run())
    assert sent[0] == b"\x00" * 640
    assert json.loads(sent[1]) == {"type": "Finalize"}
    assert json.loads(sent[2]) == {"type": "CloseStream"}


def test_shutdown_is_bounded_when_provider_never_closes(monkeypatch) -> None:
    # After CloseStream the receive loop waits for Deepgram to close; a hung
    # provider must not hold the call teardown hostage.
    import server.realtime.asr as asr_module
    monkeypatch.setattr(asr_module, "SHUTDOWN_TIMEOUT_SEC", 0.05)

    async def run() -> None:
        session = DeepgramSession("key", asyncio.Queue())
        audio: asyncio.Queue = asyncio.Queue()
        audio.put_nowait(None)  # immediate hangup
        await asyncio.wait_for(session._pump(FakeWs(hang_recv=True), audio),
                               timeout=1.0)

    asyncio.run(run())  # completing at all IS the assertion


def test_reconnect_emits_epoch_event() -> None:
    # The consumer learns the stream restarted (continuity: accumulated finals
    # survive; the next partial replaces any stale one).
    async def run() -> list:
        events: asyncio.Queue = asyncio.Queue()
        session = DeepgramSession("key", events)
        attempts = []

        async def flaky(audio, epoch=1):
            attempts.append(1)
            if len(attempts) == 1:
                from websockets.exceptions import ConnectionClosedError
                raise ConnectionClosedError(None, None)

        session._run_once = flaky
        await session.run(asyncio.Queue())
        out = []
        while not events.empty():
            out.append(events.get_nowait())
        return out

    out = asyncio.run(run())
    assert out == [AsrReconnected(epoch=2)]


def test_parser_tags_every_event_with_connection_epoch() -> None:
    assert parse_message(_results("new", False), epoch=3) == AsrPartial("new", 3)
    assert parse_message(_results("new.", True), epoch=3) == AsrFinal(
        "new.", False, 3)
