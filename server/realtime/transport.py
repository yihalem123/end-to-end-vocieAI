"""Internal audio frame contract, plus the codecs a telephony leg needs.

## How this works
Every transport normalises inbound audio to ONE internal format before it
touches the pipeline: 16 kHz, mono, PCM16, 20 ms frames = 640 bytes. The
browser leg already produces exactly that (capture-processor.js), so its
adapter is a strict size check. A telephony leg (Twilio) carries base64 mu-law
at 8 kHz in JSON "media" events, 160 bytes per 20 ms; twilio_media_to_frame()
decodes G.711 and upsamples so VAD, ASR and the endpointer never know which
leg a call came from, and frame_to_twilio_payload() is the mirror image for
outbound audio. G.711 is implemented here in numpy rather than the stdlib:
`audioop` was removed in Python 3.13. Validation is strict and cheap: a frame
of the wrong size is a protocol error, never something to pad or truncate
silently - a silently resized frame would skew every timestamp downstream.
"""
import base64

import numpy as np

FRAME_MS = 20
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2                                   # PCM16
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000     # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH         # 640

TELEPHONY_RATE = 8000
TELEPHONY_FRAME_BYTES = TELEPHONY_RATE * FRAME_MS // 1000   # 160 mu-law bytes per 20 ms

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


class FrameFormatError(ValueError):
    """Inbound payload is not one internal frame."""


def validate_frame(payload: bytes) -> bytes:
    """Accept exactly one internal frame; anything else is a protocol error."""
    if len(payload) != FRAME_BYTES:
        raise FrameFormatError(
            f"expected one {FRAME_BYTES}-byte frame "
            f"({FRAME_MS} ms PCM16 @ {SAMPLE_RATE} Hz), got {len(payload)} bytes")
    return payload


# --- G.711 mu-law -----------------------------------------------------------

def mulaw_decode(data: bytes) -> np.ndarray:
    """mu-law bytes -> int16 samples (8 kHz)."""
    u = ~np.frombuffer(data, dtype=np.uint8)
    sign = u & 0x80
    exponent = ((u >> 4) & 0x07).astype(np.int32)
    mantissa = (u & 0x0F).astype(np.int32)
    magnitude = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def mulaw_encode(samples: np.ndarray) -> bytes:
    """int16 samples (8 kHz) -> mu-law bytes."""
    pcm = samples.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0).astype(np.int32)
    pcm = np.minimum(np.abs(pcm), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.floor(np.log2(np.maximum((pcm >> 7) & 0xFF, 1))).astype(np.int32)
    mantissa = (pcm >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


# --- resampling (2x, telephony <-> internal) --------------------------------

def upsample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    """Linear interpolation: cheap, explainable, fine for speech."""
    n = len(samples)
    if n == 0:
        return samples.astype(np.int16)
    src = np.arange(n, dtype=np.float64)
    dst = np.arange(2 * n, dtype=np.float64) / 2.0
    return np.interp(dst, src, samples.astype(np.float64)).round().astype(np.int16)


def downsample_16k_to_8k(samples: np.ndarray) -> np.ndarray:
    """Average adjacent pairs: a one-tap low-pass before decimation."""
    pairs = samples.astype(np.int32)[: len(samples) // 2 * 2].reshape(-1, 2)
    return (pairs.sum(axis=1) // 2).astype(np.int16)


# --- Twilio media <-> internal frame ----------------------------------------

def twilio_media_to_frame(mulaw: bytes) -> bytes:
    """One 160-byte mu-law chunk (20 ms @ 8 kHz) -> one 640-byte internal frame."""
    if len(mulaw) != TELEPHONY_FRAME_BYTES:
        raise FrameFormatError(
            f"expected {TELEPHONY_FRAME_BYTES} mu-law bytes per 20 ms, got {len(mulaw)}")
    return validate_frame(upsample_8k_to_16k(mulaw_decode(mulaw)).tobytes())


def frame_to_twilio_payload(frame: bytes) -> str:
    """One 640-byte internal frame -> base64 of 160 mu-law bytes for a media event."""
    validate_frame(frame)
    pcm16 = np.frombuffer(frame, dtype=np.int16)
    return base64.b64encode(mulaw_encode(downsample_16k_to_8k(pcm16))).decode("ascii")
