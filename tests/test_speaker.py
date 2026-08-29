"""Sentence marks and generation ownership in speaker.py."""
import asyncio

import pytest

from server.realtime.speaker import Speaker, spoken_through


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


class FakeTts:
    async def synthesize(self, _sentence: str):
        yield bytes(1280)  # two frames


async def sentences():
    yield "First sentence."


def test_generation_invalidated_during_send_stops_further_audio() -> None:
    async def run() -> None:
        sent: list[tuple[int, bytes]] = []
        current = True

        async def send(generation_id: int, frame: bytes) -> None:
            nonlocal current
            sent.append((generation_id, frame))
            current = False

        speaker = Speaker(send, FakeTts())
        with pytest.raises(asyncio.CancelledError):
            await speaker.speak(sentences(), 0.0, 0.0, 7, lambda: current)
        assert [generation for generation, _ in sent] == [7]

    asyncio.run(run())


def test_truncation_is_scoped_to_generation_sample_count() -> None:
    async def run() -> None:
        async def send(_generation_id: int, _frame: bytes) -> None:
            pass

        speaker = Speaker(send, FakeTts())
        await speaker.speak(sentences(), 0.0, 0.0, 8, lambda: True)
        assert speaker.truncate(8, 1) == ("", -1)
        assert speaker.truncate(8, 320) == ("First sentence.", 0)
        assert speaker.truncate(99, 320) == ("", -1)

    asyncio.run(run())
