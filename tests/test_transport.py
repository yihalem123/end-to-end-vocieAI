"""The internal frame contract every transport must honour."""
import pytest

from server.realtime.transport import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    FrameFormatError,
    validate_frame,
)


def test_frame_constants_describe_20ms_of_16k_pcm16() -> None:
    assert (FRAME_MS, SAMPLE_RATE) == (20, 16000)
    assert FRAME_SAMPLES == 320
    assert FRAME_BYTES == 640          # what capture-processor.js posts


def test_exact_frame_passes_through_unchanged() -> None:
    frame = bytes(FRAME_BYTES)
    assert validate_frame(frame) is frame


@pytest.mark.parametrize("size", [0, 1, 639, 641, 1280])
def test_wrong_size_is_a_protocol_error_not_a_resize(size: int) -> None:
    with pytest.raises(FrameFormatError, match="640-byte frame"):
        validate_frame(bytes(size))
