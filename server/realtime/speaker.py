"""Speaker: streams one assistant reply to the client. Phase 3.

## How this works
speak() runs inside one supervisor-owned generation. For each sentence it records
a MARK (the frame offset where that sentence's audio starts), opens an
ElevenLabs stream (one per sentence — the reconnect cost is measured, and is the
known refinement target), reslices arriving PCM into 640-byte frames, and sends
them FRAME-PACED: an initial PREBUFFER_FRAMES burst absorbs network jitter, then
one frame per 20 ms of a deadline schedule (sleep-until-target, so pacing never
drifts). Pacing keeps the client's buffer shallow, which is what makes sent ≈
played and barge-in truncation honest.

Barge-in invalidates the generation before cancellation. Checks before every TTS
sentence and paced audio send prevent stale work from crossing the WebSocket.
The client replies with generation id + total played sample count. truncate()
  maps that generation-relative count onto its immutable marks, then
  spoken_through() finds the last sentence that got any airtime. That spoken
  prefix is what actually reached the caller's ears — the truthful transcript.

Timings returned per reply: tts_ttfb (commit -> first ElevenLabs byte),
first_audio (commit -> first frame sent), turn_latency (vad_stop -> first frame
sent: what the caller experienced as "the agent thought about it").
"""
import asyncio
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Callable

from server.realtime.tts import FRAME_BYTES, FrameChunker

SAMPLES_PER_FRAME = FRAME_BYTES // 2
FRAME_SEC = 0.02
PREBUFFER_FRAMES = 10  # 200 ms head start before pacing applies


def spoken_through(marks: list[int], played_frames: int) -> int:
    """Index of the last sentence that got any airtime; -1 if none did."""
    idx = -1
    for i, mark in enumerate(marks):
        if played_frames > mark:
            idx = i
    return idx


class Speaker:
    def __init__(self, send_audio, tts) -> None:  # tts: anything with .synthesize()
        self._send_audio = send_audio
        self._tts = tts
        self.samples_sent_total = 0  # observability only; acks are per generation
        self._records: dict[int, PlaybackRecord] = {}

    async def speak(self, sentences: AsyncIterable[str], commit_t: float,
                    vad_stop_t: float, generation_id: int,
                    is_current: Callable[[], bool]) -> dict:
        # Streaming input: sentences arrive as the LLM writes them, so sentence
        # one is playing while sentence three is still being generated.
        record = PlaybackRecord()
        self._records[generation_id] = record
        self._prune_records()
        timings: dict[str, float] = {}
        pace_start: float | None = None
        frames_sent = 0
        try:
            async for sentence in sentences:
                self._require_current(is_current)
                record.sentences.append(sentence)
                record.marks.append(frames_sent)
                chunker = FrameChunker()
                async for chunk in self._tts.synthesize(sentence):
                    self._require_current(is_current)
                    if "tts_ttfb_ms" not in timings:
                        timings["tts_ttfb_ms"] = (time.monotonic() - commit_t) * 1000
                    for frame in chunker.push(chunk):
                        if pace_start is None:
                            pace_start = time.monotonic()
                            timings["first_audio_ms"] = (pace_start - commit_t) * 1000
                            timings["turn_latency_ms"] = (pace_start - vad_stop_t) * 1000
                        frames_sent = await self._send_paced(
                            frame, pace_start, frames_sent, generation_id, is_current)
                tail = chunker.flush()
                if tail is not None and pace_start is not None:
                    frames_sent = await self._send_paced(
                        tail, pace_start, frames_sent, generation_id, is_current)
            return timings
        finally:
            # On cancellation (barge-in, hangup) partial timings still matter:
            # a turn that got its first byte out has a measurable ttfb.
            timings.setdefault("interrupted", float(frames_sent))

    async def _send_paced(self, frame: bytes, pace_start: float, frames_sent: int,
                          generation_id: int,
                          is_current: Callable[[], bool]) -> int:
        target = pace_start + max(0, frames_sent - PREBUFFER_FRAMES) * FRAME_SEC
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._require_current(is_current)
        await self._send_audio(generation_id, frame)
        self.samples_sent_total += SAMPLES_PER_FRAME
        self._require_current(is_current)
        return frames_sent + 1

    def text(self, generation_id: int) -> str:
        record = self._records.get(generation_id)
        return "" if record is None else " ".join(record.sentences)

    def truncate(self, generation_id: int,
                 client_played_samples: int) -> tuple[str, int]:
        """Map playback position onto one specific generation's sentence marks."""
        record = self._records.get(generation_id)
        if record is None:
            return "", -1
        played_frames = max(0, client_played_samples) // SAMPLES_PER_FRAME
        idx = spoken_through(record.marks, played_frames)
        return " ".join(record.sentences[: idx + 1]), idx

    @staticmethod
    def _require_current(is_current: Callable[[], bool]) -> None:
        if not is_current():
            raise asyncio.CancelledError

    def _prune_records(self) -> None:
        while len(self._records) > 4:
            del self._records[next(iter(self._records))]


@dataclass
class PlaybackRecord:
    sentences: list[str] = field(default_factory=list)
    marks: list[int] = field(default_factory=list)
