"""Sentence marks and generation ownership in speaker.py."""
import asyncio
import time

import pytest

from server.realtime.speaker import Speaker, playback_remaining_seconds, spoken_through


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
            await speaker.speak(sentences(), {"commit_t": 0.0, "vad_stop_t": 0.0}, 7, lambda: current)
        assert [generation for generation, _ in sent] == [7]

    asyncio.run(run())


def test_truncation_is_scoped_to_generation_sample_count() -> None:
    async def run() -> None:
        async def send(_generation_id: int, _frame: bytes) -> None:
            pass

        speaker = Speaker(send, FakeTts())
        await speaker.speak(sentences(), {"commit_t": 0.0, "vad_stop_t": 0.0}, 8, lambda: True)
        assert speaker.truncate(8, 1) == ("", -1)
        assert speaker.truncate(8, 320) == ("First sentence.", 0)
        assert speaker.truncate(99, 320) == ("", -1)

    asyncio.run(run())


def test_playback_remaining_uses_sent_frames_and_elapsed_time() -> None:
    assert playback_remaining_seconds(0, None, now=10.0) == 0.0
    assert playback_remaining_seconds(20, 10.0, now=10.1) == pytest.approx(0.3)
    assert playback_remaining_seconds(20, 10.0, now=11.0) == 0.0


def test_synthesis_prefetches_next_sentence_while_first_plays() -> None:
    class RecordingTts:
        def __init__(self):
            self.calls = []

        async def synthesize(self, sentence):
            self.calls.append(sentence)
            yield bytes(640 * 20)

    async def two_sentences():
        yield "First."
        yield "Second."

    async def run() -> None:
        tts = RecordingTts()

        async def send(_generation, _frame):
            await asyncio.sleep(0.002)

        speaker = Speaker(send, tts)
        now = time.monotonic()
        await speaker.speak(
            two_sentences(),
            {"commit_t": now, "vad_stop_t": now,
             "generation_start_t": now},
            4, lambda: True)
        assert tts.calls == ["First.", "Second."]
        assert speaker._records[4].marks == [0, 20]

    asyncio.run(run())


def test_tts_first_byte_is_measured_before_commit_gated_audio() -> None:
    async def run() -> None:
        sent = []
        release = asyncio.Event()

        async def send(_generation, frame):
            sent.append(frame)

        speaker = Speaker(send, FakeTts())
        started = time.monotonic()
        task = asyncio.create_task(speaker.speak(
            sentences(),
            {"commit_t": started, "vad_stop_t": started,
             "generation_start_t": started},
            5, lambda: True, release=release))
        await asyncio.sleep(0.01)
        assert sent == []
        release.set()
        timings = await task
        assert timings["tts_ttfb_ms"] < timings["first_audio_ms"]

    asyncio.run(run())
