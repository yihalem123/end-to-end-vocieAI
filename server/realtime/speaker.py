"""Speaker: streams one assistant reply to the client. Phase 3.

## How this works
speak() runs as its own cancellable task per reply. For each sentence it records
a MARK (the frame offset where that sentence's audio starts), opens an
ElevenLabs stream (one per sentence — the reconnect cost is measured, and is the
known refinement target), reslices arriving PCM into 640-byte frames, and sends
them FRAME-PACED: an initial PREBUFFER_FRAMES burst absorbs network jitter, then
one frame per 20 ms of a deadline schedule (sleep-until-target, so pacing never
drifts). Pacing keeps the client's buffer shallow, which is what makes sent ≈
played and barge-in truncation honest.

Barge-in path: call.py cancels this task (CancelledError re-raised per CLAUDE.md)
and tells the client to flush; the client replies with its total played sample
count. truncate() maps that onto the marks: played frames of THIS reply =
(client total played) - (samples sent before this reply began), then
spoken_through() finds the last sentence that got any airtime. That spoken
prefix is what actually reached the caller's ears — the truthful transcript.

Timings returned per reply: tts_ttfb (commit -> first ElevenLabs byte),
first_audio (commit -> first frame sent), turn_latency (vad_stop -> first frame
sent: what the caller experienced as "the agent thought about it").
"""
import asyncio
import time
from collections.abc import AsyncIterable

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
    def __init__(self, send_bytes, tts) -> None:  # tts: anything with .synthesize()
        self._send_bytes = send_bytes
        self._tts = tts
        self.samples_sent_total = 0  # lifetime, matches the client's played counter
        self.reply_base = 0          # samples_sent_total when current reply began
        self.sentences: list[str] = []
        self.marks: list[int] = []   # frame offset (within reply) per sentence

    async def speak(self, sentences: AsyncIterable[str], commit_t: float,
                    vad_stop_t: float) -> dict:
        # Streaming input: sentences arrive as the LLM writes them, so sentence
        # one is playing while sentence three is still being generated.
        self.sentences = []
        self.marks = []
        self.reply_base = self.samples_sent_total
        timings: dict[str, float] = {}
        pace_start: float | None = None
        frames_sent = 0
        try:
            async for sentence in sentences:
                self.sentences.append(sentence)
                self.marks.append(frames_sent)
                chunker = FrameChunker()
                async for chunk in self._tts.synthesize(sentence):
                    if "tts_ttfb_ms" not in timings:
                        timings["tts_ttfb_ms"] = (time.monotonic() - commit_t) * 1000
                    for frame in chunker.push(chunk):
                        if pace_start is None:
                            pace_start = time.monotonic()
                            timings["first_audio_ms"] = (pace_start - commit_t) * 1000
                            timings["turn_latency_ms"] = (pace_start - vad_stop_t) * 1000
                        frames_sent = await self._send_paced(frame, pace_start, frames_sent)
                tail = chunker.flush()
                if tail is not None and pace_start is not None:
                    frames_sent = await self._send_paced(tail, pace_start, frames_sent)
            return timings
        finally:
            # On cancellation (barge-in, hangup) partial timings still matter:
            # a turn that got its first byte out has a measurable ttfb.
            timings.setdefault("interrupted", float(frames_sent))

    async def _send_paced(self, frame: bytes, pace_start: float, frames_sent: int) -> int:
        target = pace_start + max(0, frames_sent - PREBUFFER_FRAMES) * FRAME_SEC
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await self._send_bytes(frame)
        self.samples_sent_total += SAMPLES_PER_FRAME
        return frames_sent + 1

    def truncate(self, client_played_samples: int) -> tuple[str, int]:
        """Map the client's played counter to the sentences actually heard."""
        played_frames = max(0, client_played_samples - self.reply_base) // SAMPLES_PER_FRAME
        idx = spoken_through(self.marks, played_frames)
        return " ".join(self.sentences[: idx + 1]), idx
