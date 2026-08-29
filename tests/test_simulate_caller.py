"""Headless live-harness coverage for the generation-aware playback protocol."""

from scripts.simulate_caller import FRAME_BYTES, PlaybackTracker
from server.realtime.protocol import encode_audio_frame


def test_simulator_decodes_audio_and_acks_drain_for_its_generation() -> None:
    tracker = PlaybackTracker()
    generation_id, pcm = tracker.on_audio(encode_audio_frame(12, bytes(FRAME_BYTES)))

    assert generation_id == 12
    assert pcm == bytes(FRAME_BYTES)
    assert tracker.acknowledgement({
        "type": "audio_end", "generation_id": 12,
    }) == {"type": "playback_drained", "generation_id": 12}


def test_simulator_acks_clear_with_generation_relative_played_samples() -> None:
    tracker = PlaybackTracker()
    tracker.on_audio(encode_audio_frame(3, bytes(FRAME_BYTES)))
    tracker.on_audio(encode_audio_frame(3, bytes(FRAME_BYTES)))

    assert tracker.acknowledgement({"type": "clear", "generation_id": 3}) == {
        "type": "cleared", "generation_id": 3, "played_samples": 640,
    }
    assert tracker.acknowledgement({"type": "clear"}) is None
