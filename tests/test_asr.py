"""Deepgram URL construction and message parsing (pure parts of asr.py)."""
import json
from urllib.parse import parse_qs, urlparse

from server.realtime.asr import AsrFinal, AsrPartial, AsrUtteranceEnd, build_url, parse_message


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
