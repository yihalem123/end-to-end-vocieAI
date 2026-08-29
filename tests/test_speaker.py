"""Sentence-mark truncation math for barge-in (pure part of speaker.py)."""
from server.realtime.speaker import spoken_through


def test_nothing_played_means_nothing_spoken() -> None:
    assert spoken_through(marks=[0, 50, 120], played_frames=0) == -1


def test_mid_first_sentence() -> None:
    # Playback stopped inside sentence 0: it was audible, so it counts.
    assert spoken_through(marks=[0, 50, 120], played_frames=30) == 0


def test_boundary_is_exclusive() -> None:
    # Stopped exactly where sentence 1 starts: not a single frame of it played.
    assert spoken_through(marks=[0, 50, 120], played_frames=50) == 0


def test_mid_last_sentence() -> None:
    assert spoken_through(marks=[0, 50, 120], played_frames=121) == 2


def test_single_sentence_reply() -> None:
    assert spoken_through(marks=[0], played_frames=1) == 0
