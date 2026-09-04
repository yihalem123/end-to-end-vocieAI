"""Internal audio frame contract shared by every transport.

## How this works
Every transport normalises inbound audio to ONE internal format before it
touches the pipeline: 16 kHz, mono, PCM16, 20 ms frames = 640 bytes. The
browser leg already produces exactly that (capture-processor.js), so its
adapter is a strict size check. A telephony leg (Twilio) carries base64 mu-law
at 8 kHz in JSON "media" events, 160 bytes per 20 ms; its adapter decodes and
upsamples to the same 640-byte frame so VAD, ASR and the endpointer never know
which leg a call came from. Outbound audio is the mirror image: the pipeline
emits 16 kHz PCM16 and each transport re-encodes for its own wire. Validation
is strict and cheap: a frame of the wrong size is a protocol error, never
something to pad or truncate silently - a silently resized frame would skew
every timestamp downstream.
"""
FRAME_MS = 20
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2                                   # PCM16
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000     # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH         # 640


class FrameFormatError(ValueError):
    """Inbound payload is not one internal frame."""


def validate_frame(payload: bytes) -> bytes:
    """Accept exactly one internal frame; anything else is a protocol error."""
    if len(payload) != FRAME_BYTES:
        raise FrameFormatError(
            f"expected one {FRAME_BYTES}-byte frame "
            f"({FRAME_MS} ms PCM16 @ {SAMPLE_RATE} Hz), got {len(payload)} bytes")
    return payload
