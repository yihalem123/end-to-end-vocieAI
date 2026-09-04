"""The internal frame contract every transport must honour, and the telephony codecs."""
import base64

import numpy as np
import pytest

from server.realtime.transport import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TELEPHONY_FRAME_BYTES,
    FrameFormatError,
    downsample_16k_to_8k,
    frame_to_twilio_payload,
    mulaw_decode,
    mulaw_encode,
    twilio_media_to_frame,
    upsample_8k_to_16k,
    validate_frame,
)


def test_frame_constants_describe_20ms_of_16k_pcm16() -> None:
    assert (FRAME_MS, SAMPLE_RATE) == (20, 16000)
    assert FRAME_SAMPLES == 320
    assert FRAME_BYTES == 640          # what capture-processor.js posts
    assert TELEPHONY_FRAME_BYTES == 160  # what Twilio sends per 20 ms


def test_exact_frame_passes_through_unchanged() -> None:
    frame = bytes(FRAME_BYTES)
    assert validate_frame(frame) is frame


@pytest.mark.parametrize("size", [0, 1, 639, 641, 1280])
def test_wrong_size_is_a_protocol_error_not_a_resize(size: int) -> None:
    with pytest.raises(FrameFormatError, match="640-byte frame"):
        validate_frame(bytes(size))


# --- G.711 mu-law ---------------------------------------------------------

def test_mulaw_known_vectors() -> None:
    # G.711 extremes: 0x00 / 0x80 are the largest magnitudes, 0xFF / 0x7F are zero.
    assert mulaw_decode(bytes([0x00, 0x80, 0xFF, 0x7F])).tolist() == [-32124, 32124, 0, 0]
    assert mulaw_encode(np.array([32124, -32124, 0], dtype=np.int16)) == bytes([0x80, 0x00, 0xFF])


def test_mulaw_round_trip_keeps_speech_shaped_signal() -> None:
    t = np.arange(800) / 8000.0
    tone = (8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    back = mulaw_decode(mulaw_encode(tone))
    err = np.abs(back.astype(np.int32) - tone.astype(np.int32))
    assert err.max() < 520          # one mu-law quantisation step at this level
    assert err.mean() < 120
    assert np.corrcoef(back, tone)[0, 1] > 0.999


# --- resampling ----------------------------------------------------------

def test_resamplers_change_length_by_exactly_two() -> None:
    eight = np.arange(160, dtype=np.int16)
    sixteen = upsample_8k_to_16k(eight)
    assert len(sixteen) == 320
    assert sixteen[::2].tolist() == eight.tolist()      # originals preserved
    assert sixteen[1] == 0 or sixteen[1] == 1           # midpoint interpolated
    assert len(downsample_16k_to_8k(sixteen)) == 160


def test_downsample_of_upsample_is_close_to_identity() -> None:
    t = np.arange(160) / 8000.0
    tone = (6000 * np.sin(2 * np.pi * 400 * t)).astype(np.int16)
    back = downsample_16k_to_8k(upsample_8k_to_16k(tone))
    assert np.abs(back.astype(np.int32) - tone.astype(np.int32)).max() < 600


# --- Twilio media <-> internal frame -------------------------------------

def test_one_twilio_chunk_becomes_one_internal_frame() -> None:
    frame = twilio_media_to_frame(bytes([0xFF]) * 160)   # 20 ms of silence
    assert len(frame) == FRAME_BYTES
    assert not any(frame)


def test_short_twilio_chunk_is_a_protocol_error() -> None:
    with pytest.raises(FrameFormatError, match="160 mu-law bytes"):
        twilio_media_to_frame(bytes(159))


def test_internal_frame_round_trips_through_the_phone_leg() -> None:
    t = np.arange(FRAME_SAMPLES) / 16000.0
    frame = (5000 * np.sin(2 * np.pi * 350 * t)).astype(np.int16).tobytes()
    payload = frame_to_twilio_payload(frame)
    mulaw = base64.b64decode(payload)
    assert len(mulaw) == TELEPHONY_FRAME_BYTES
    back = np.frombuffer(twilio_media_to_frame(mulaw), dtype=np.int16)
    original = np.frombuffer(frame, dtype=np.int16)
    assert np.corrcoef(back, original)[0, 1] > 0.99
