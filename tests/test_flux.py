"""Flux (Deepgram v2) URL construction and TurnInfo parsing."""
import json
from urllib.parse import parse_qs, urlparse

from server.realtime.flux import (
    FluxEndOfTurn,
    FluxEagerEndOfTurn,
    FluxStartOfTurn,
    FluxTurnResumed,
    FluxUpdate,
    build_flux_url,
    parse_flux_message,
)


def test_build_flux_url_pins_model_and_audio() -> None:
    url = build_flux_url()
    parsed = urlparse(url)
    assert parsed.path == "/v2/listen"
    q = parse_qs(parsed.query)
    assert q["model"] == ["flux-general-en"]
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == ["16000"]
    assert q["eot_threshold"] == ["0.7"]
    assert q["eager_eot_threshold"] == ["0.6"]


def _turn_info(event: str, transcript: str = "") -> str:
    return json.dumps({"type": "TurnInfo", "event": event,
                       "transcript": transcript, "end_of_turn_confidence": 0.9})


def test_parse_end_of_turn() -> None:
    ev = parse_flux_message(_turn_info("EndOfTurn", "I have five years."))
    assert ev == FluxEndOfTurn(transcript="I have five years.")


def test_parse_update_is_a_partial() -> None:
    ev = parse_flux_message(_turn_info("Update", "I have"))
    assert ev == FluxUpdate(transcript="I have")


def test_parse_start_of_turn() -> None:
    assert parse_flux_message(_turn_info("StartOfTurn")) == FluxStartOfTurn()


def test_parse_ignores_connected_and_unknown() -> None:
    assert parse_flux_message(json.dumps({"type": "Connected"})) is None
    assert parse_flux_message(_turn_info("OtherFutureEvent")) is None
    assert parse_flux_message("not json") is None


def test_parse_eager_lifecycle_and_epoch() -> None:
    assert parse_flux_message(_turn_info("EagerEndOfTurn", "Done."), 4) == (
        FluxEagerEndOfTurn("Done.", 4))
    assert parse_flux_message(_turn_info("TurnResumed"), 4) == FluxTurnResumed(4)
