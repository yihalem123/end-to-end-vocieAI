"""Generation-aware server-to-browser audio wire format.

## How this works
Inbound browser audio remains raw 640-byte PCM16. Outbound agent audio prefixes
each 640-byte frame with an unsigned 64-bit network-order generation id. The
browser strips the eight-byte header before playback. JSON control messages use
the same integer generation id for clear, cleared, audio_end, playback_drained
and playback_overflow, so delayed control or audio from an older generation can
be rejected deterministically.
"""
import struct

PCM_FRAME_BYTES = 640
AUDIO_HEADER = struct.Struct("!Q")
AUDIO_WIRE_BYTES = AUDIO_HEADER.size + PCM_FRAME_BYTES


def encode_audio_frame(generation_id: int, pcm: bytes) -> bytes:
    if generation_id <= 0:
        raise ValueError("generation_id must be positive")
    if len(pcm) != PCM_FRAME_BYTES:
        raise ValueError(f"audio frame must be {PCM_FRAME_BYTES} bytes")
    return AUDIO_HEADER.pack(generation_id) + pcm


def decode_audio_frame(data: bytes) -> tuple[int, bytes]:
    if len(data) != AUDIO_WIRE_BYTES:
        raise ValueError(f"wire frame must be {AUDIO_WIRE_BYTES} bytes")
    return AUDIO_HEADER.unpack(data[:AUDIO_HEADER.size])[0], data[AUDIO_HEADER.size:]
