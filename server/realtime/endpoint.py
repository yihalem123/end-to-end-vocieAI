"""Endpointer: decides WHEN the caller's turn is over. Phase 2.

## How this works
The hardest problem in voice UX: commit too early and you interrupt mid-thought
("I worked at… um…"); too late and the agent feels dead. This is a synchronous
state machine — every method takes an explicit timestamp and returns a value, no
sleeps, no tasks — so the whole policy is unit-testable with fake clocks; the
caller (call.py) drives tick() from a ~50 ms loop with real time.

Policy (from PLAN.md): when VAD reports silence (on_vad_stop) we arm a deadline —
vad_stop + 250 ms if the accumulated transcript "looks complete" (ends in
terminal punctuation), else vad_stop + 1200 ms of extra patience. Each ASR final
re-arms the deadline from the SAME vad_stop anchor, because finals lag speech: a
punctuated final arriving 150 ms after silence upgrades the pending slow wait to
the fast one without restarting the clock. Deepgram's UtteranceEnd commits
immediately (its own word-gap evidence). VAD start disarms everything — the
caller resumed. Commits require non-empty text (a cough triggers VAD, not a turn)
and reset state for the next turn. endpoint_delay = commit time - vad_stop: the
per-turn cost of this policy, our first-class metric.
"""
from dataclasses import dataclass

_TERMINAL = (".", "!", "?")


def looks_complete(text: str) -> bool:
    return text.rstrip().endswith(_TERMINAL)


@dataclass(frozen=True)
class TurnComplete:
    transcript: str
    endpoint_delay: float  # seconds from vad_stop to commit
    vad_stop_t: float
    commit_t: float
    reason: str  # "fast" | "slow" | "utterance_end"


class Endpointer:
    def __init__(self, fast_sec: float = 0.25, slow_sec: float = 1.20) -> None:
        self.fast_sec = fast_sec
        self.slow_sec = slow_sec
        self._finals: list[str] = []
        self._vad_stop_t: float | None = None  # None = speaking or idle, nothing armed

    @property
    def transcript(self) -> str:
        return " ".join(self._finals)

    def _reset(self) -> None:
        self._finals = []
        self._vad_stop_t = None

    def _commit(self, t: float, reason: str) -> TurnComplete | None:
        if not self.transcript or self._vad_stop_t is None:
            return None
        turn = TurnComplete(
            transcript=self.transcript,
            endpoint_delay=t - self._vad_stop_t,
            vad_stop_t=self._vad_stop_t,
            commit_t=t,
            reason=reason,
        )
        self._reset()
        return turn

    def _deadline(self) -> float | None:
        if self._vad_stop_t is None:
            return None
        patience = self.fast_sec if looks_complete(self.transcript) else self.slow_sec
        return self._vad_stop_t + patience

    # --- events ---

    def on_vad_start(self, t: float) -> None:
        self._vad_stop_t = None  # caller resumed: disarm any pending commit

    def on_vad_stop(self, t: float) -> None:
        self._vad_stop_t = t

    def on_asr_final(self, text: str, t: float) -> None:
        text = text.strip()
        if text:
            self._finals.append(text)
        # No deadline bookkeeping needed: _deadline() recomputes completeness
        # from the current transcript, still anchored at vad_stop.

    def on_utterance_end(self, t: float) -> TurnComplete | None:
        return self._commit(t, "utterance_end")

    def tick(self, t: float) -> TurnComplete | None:
        deadline = self._deadline()
        if deadline is None or t < deadline:
            return None
        reason = "fast" if looks_complete(self.transcript) else "slow"
        return self._commit(t, reason)
