"""Browser control messages parse into exactly one typed event or None."""
import json

import pytest

from server.realtime.client_events import (
    ClientChat,
    ClientCleared,
    ClientPlaybackDrained,
    ClientPlaybackOverflow,
    ClientPlaybackStarted,
    parse_client_message,
)


def _msg(**fields) -> str:
    return json.dumps(fields)


def test_acks_carry_their_generation_id() -> None:
    assert parse_client_message(_msg(type="cleared", generation_id=4, played_samples=320)) \
        == ClientCleared(4, 320)
    assert parse_client_message(_msg(type="playback_drained", generation_id=4)) \
        == ClientPlaybackDrained(4)
    assert parse_client_message(_msg(type="playback_started", generation_id=4)) \
        == ClientPlaybackStarted(4)
    assert parse_client_message(_msg(type="playback_overflow", generation_id=4,
                                     played_samples=8)) == ClientPlaybackOverflow(4, 8)


def test_played_samples_are_clamped_at_zero() -> None:
    assert parse_client_message(_msg(type="cleared", generation_id=1, played_samples=-7)) \
        == ClientCleared(1, 0)


@pytest.mark.parametrize("gen", [0, -1, "0"])
def test_a_non_positive_generation_id_is_not_an_ack(gen) -> None:
    assert parse_client_message(_msg(type="cleared", generation_id=gen)) is None


def test_chat_text_is_trimmed_and_empty_chat_is_dropped() -> None:
    assert parse_client_message(_msg(type="chat", text="  yes  ")) == ClientChat("yes")
    assert parse_client_message(_msg(type="chat", text="   ")) is None


@pytest.mark.parametrize("raw", [
    "not json",
    "[1, 2, 3]",                                       # JSON, but not an object
    _msg(type="cleared", generation_id="four"),        # int() fails -> None
    _msg(type="cleared", generation_id=[1]),           # TypeError -> None
    _msg(type="teleport", generation_id=1),            # unknown type
    _msg(),                                            # no type at all
])
def test_a_confused_client_can_never_kill_the_call(raw: str) -> None:
    assert parse_client_message(raw) is None
