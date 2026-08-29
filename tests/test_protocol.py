"""Generation-prefixed binary audio protocol."""
import pytest

from server.realtime.protocol import (
    AUDIO_WIRE_BYTES,
    PCM_FRAME_BYTES,
    decode_audio_frame,
    encode_audio_frame,
)


def test_audio_frame_round_trip_keeps_generation_and_pcm() -> None:
    pcm = bytes(range(256)) * 2 + bytes(128)
    wire = encode_audio_frame(42, pcm)
    assert len(wire) == AUDIO_WIRE_BYTES
    assert decode_audio_frame(wire) == (42, pcm)


def test_audio_protocol_rejects_invalid_identity_and_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        encode_audio_frame(0, bytes(PCM_FRAME_BYTES))
    with pytest.raises(ValueError, match="640"):
        encode_audio_frame(1, b"short")
    with pytest.raises(ValueError, match="648"):
        decode_audio_frame(b"short")
