"""ElevenLabs stream-input protocol: URL, message parsing, frame chunking."""
import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

from server.realtime.tts import (
    FrameChunker,
    MultiContextTts,
    TtsBufferOverflow,
    build_multi_url,
    build_url,
    parse_multi_message,
    parse_tts_message,
)


def test_build_url_pins_model_and_format() -> None:
    url = build_url(voice_id="test-voice")
    parsed = urlparse(url)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.elevenlabs.io"
    assert parsed.path == "/v1/text-to-speech/test-voice/stream-input"
    q = parse_qs(parsed.query)
    assert q["model_id"] == ["eleven_flash_v2_5"]  # lowest-latency model
    assert q["output_format"] == ["pcm_16000"]     # matches our internal format


def test_parse_audio_message() -> None:
    pcm = b"\x01\x02" * 20
    raw = json.dumps({"audio": base64.b64encode(pcm).decode(), "isFinal": None})
    audio, is_final = parse_tts_message(raw)
    assert audio == pcm
    assert is_final is False


def test_parse_final_message() -> None:
    audio, is_final = parse_tts_message(json.dumps({"isFinal": True}))
    assert audio == b""
    assert is_final is True


def test_parse_empty_audio_field() -> None:
    # ElevenLabs sends alignment-only messages with audio null or absent.
    audio, is_final = parse_tts_message(json.dumps({"audio": None, "isFinal": None}))
    assert audio == b""
    assert is_final is False


def test_build_multi_url_pins_agent_options() -> None:
    url = build_multi_url(voice_id="v1")
    parsed = urlparse(url)
    assert parsed.path == "/v1/text-to-speech/v1/multi-stream-input"
    q = parse_qs(parsed.query)
    assert q["model_id"] == ["eleven_flash_v2_5"]
    assert q["output_format"] == ["pcm_16000"]
    assert q["auto_mode"] == ["true"]           # full sentences: skip chunk buffering
    assert q["inactivity_timeout"] == ["180"]   # survive silences between replies


def test_parse_multi_message_routes_by_context() -> None:
    pcm = b"\x0a\x0b" * 10
    raw = json.dumps({"audio": base64.b64encode(pcm).decode(), "contextId": "c3"})
    ctx, audio, is_final = parse_multi_message(raw)
    assert (ctx, audio, is_final) == ("c3", pcm, False)


def test_parse_multi_final_message() -> None:
    ctx, audio, is_final = parse_multi_message(
        json.dumps({"isFinal": True, "contextId": "c3"}))
    assert (ctx, audio, is_final) == ("c3", b"", True)


def test_frame_chunker_reslices_to_640_bytes() -> None:
    # ElevenLabs chunks arrive at arbitrary sizes; the wire wants 640-byte
    # frames. Odd remainders must carry over, nothing dropped or padded early.
    chunker = FrameChunker()
    frames = list(chunker.push(b"a" * 1000))
    assert frames == [b"a" * 640]                 # 360 bytes held back
    frames = list(chunker.push(b"b" * 1000))      # 360+1000 = 1360 -> 2 frames + 80
    assert len(frames) == 2
    assert all(len(f) == 640 for f in frames)
    assert frames[0] == b"a" * 360 + b"b" * 280   # carry precedes new bytes
    tail = chunker.flush()
    assert tail is not None and len(tail) == 640  # remainder zero-padded to a full frame
    assert tail[:80] == b"b" * 80
    assert tail[80:] == bytes(560)


def test_frame_chunker_flush_empty_returns_none() -> None:
    chunker = FrameChunker()
    list(chunker.push(b"x" * 640))
    assert chunker.flush() is None                # nothing pending, no phantom frame


def test_multi_context_queue_overflow_fails_closed() -> None:
    tts = MultiContextTts("key", "voice")
    queue = asyncio.Queue(maxsize=2)
    tts._offer_context(queue, (b"first", False))
    tts._offer_context(queue, (b"second", False))
    tts._offer_context(queue, (b"overflow", False))
    item = queue.get_nowait()
    assert isinstance(item, TtsBufferOverflow)
    assert queue.empty()  # buffered stale audio was discarded, not replayed later
