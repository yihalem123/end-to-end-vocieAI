"""Endpointer: decides WHEN the caller's turn is over. Phase 2.

## How this works
The hardest problem in voice UX: commit too early and you interrupt mid-thought
("I worked at… um…"); too late and the agent feels dead. This is a synchronous
state machine — every method takes an explicit timestamp and returns a value, no
sleeps, no tasks — so the whole policy is unit-testable with fake clocks; the
caller (call.py) drives tick() from a ~50 ms loop with real time.

Policy: when VAD reports silence (on_vad_stop) we arm a deadline anchored at
vad_stop, with THREE patience tiers judged from the accumulated transcript:
- complete (ends in terminal punctuation)            -> +250 ms   ("fast")
- incomplete                                          -> +2.0 s    ("slow")
- trailing (ends mid-clause: a comma/semicolon, or a
  conjunction/preposition/filler like "and", "to",
  "um" — the thought is clearly still in flight)      -> +2.5 s    ("trailing")
The slow tier is deliberately generous: Deepgram punctuates finished speech, so
an UNpunctuated ending usually means its prosody model heard continuation too.
Snappiness lives in the fast tier; only ambiguity pays for patience.
Each ASR final re-arms the deadline from the SAME vad_stop anchor, because
finals lag speech: a punctuated final arriving 150 ms after silence upgrades a
pending slow wait to fast without restarting the clock. Deepgram's UtteranceEnd
(1 s word gap) commits immediately ONLY when the text looks complete — it is
evidence, not a command; tuned 2026-08-30 after live turns like "…though. And"
were cut off 157 ms after silence. The trailing-word check is the cheap lexical
version of semantic turn detection (the managed alternative: Deepgram Flux's
model-integrated end-of-turn). VAD start disarms everything — the caller
resumed. Commits require non-empty text (a cough triggers VAD, not a turn) and
reset state for the next turn. endpoint_delay = commit time - vad_stop: the
per-turn cost of this policy, our first-class metric.
"""
from dataclasses import dataclass

_TERMINAL = (".", "!", "?")
_TRAILING_WORDS = frozenset(
    "and but or so to of in on at with for from the a an um uh like because "
    "if when while my your our their his her its i".split()
)


def looks_complete(text: str) -> bool:
    return text.rstrip().endswith(_TERMINAL)


def looks_trailing(text: str) -> bool:
    text = text.rstrip()
    if text.endswith((",", ";", ":")):
        return True  # mid-clause punctuation: the strongest "still going" cue
    words = text.split()
    return bool(words) and words[-1].lower() in _TRAILING_WORDS


@dataclass(frozen=True)
class TurnComplete:
    transcript: str
    endpoint_delay: float  # seconds from vad_stop to commit
    vad_stop_t: float
    commit_t: float
    reason: str  # "fast" | "slow" | "trailing" | "utterance_end"


class Endpointer:
    def __init__(self, fast_sec: float = 0.25, slow_sec: float = 2.00,
                 trailing_sec: float = 2.50) -> None:
        self.fast_sec = fast_sec
        self.slow_sec = slow_sec
        self.trailing_sec = trailing_sec
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

    def _tier(self) -> tuple[float, str]:
        text = self.transcript
        if looks_complete(text):
            return self.fast_sec, "fast"
        if looks_trailing(text):
            return self.trailing_sec, "trailing"
        return self.slow_sec, "slow"

    def _deadline(self) -> float | None:
        if self._vad_stop_t is None:
            return None
        patience, _ = self._tier()
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
        # Evidence, not a command: a 1 s word gap on a clearly unfinished
        # thought ("…though. And") means thinking, not done. The timers decide.
        if not looks_complete(self.transcript):
            return None
        return self._commit(t, "utterance_end")

    def tick(self, t: float) -> TurnComplete | None:
        deadline = self._deadline()
        if deadline is None or t < deadline:
            return None
        _, reason = self._tier()
        return self._commit(t, reason)
