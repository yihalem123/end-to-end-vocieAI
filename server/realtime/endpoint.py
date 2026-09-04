"""Endpointer: decides WHEN the caller's turn is over. Phase 2.

## How this works
The hardest problem in voice UX: commit too early and you interrupt mid-thought
("I worked at… um…"); too late and the agent feels dead. This is a synchronous
state machine — every method takes an explicit timestamp and returns a value, no
sleeps, no tasks — so the whole policy is unit-testable with fake clocks; the
caller (call.py) drives tick() from a ~50 ms loop with real time.

Policy: when VAD reports silence (on_vad_stop) we arm a deadline anchored at
vad_stop, with THREE patience tiers judged from the accumulated transcript:
- complete (ends in terminal punctuation)            -> +200 ms   ("fast")
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
    "if when while my your our their his her its i "
    # Copulas/auxiliaries: unpunctuated "...my license is" is nearly always a
    # thought in flight. (Punctuated finals take the fast tier before this
    # check, so "I said that." is unaffected.) Found by the simulated caller.
    "is are was were am be been being has have had will would shall can "
    "could should may might must do does did that".split()
)

_TRAILING_CONTRACTIONS = frozenset({
    "i'm", "im", "it's", "its", "we're", "were", "they're", "theyre",
    "you're", "youre", "i've", "ive", "we've", "weve", "i'd", "id",
    "we'd", "wed", "can't", "cant", "don't", "dont", "that's", "thats",
    "there's", "theres", "he's", "hes", "she's", "shes",
})


_EDGE_PUNCT = '"' + "'" + "()[]{}"  # quote/bracket edges ASR sometimes attaches
# Words that cannot end a sentence even when the ASR puts a period after
# them. Deliberately NARROWER than _TRAILING_WORDS: copulas and modals are
# trailing when unpunctuated ("my license is") but complete short answers
# when punctuated ("Yes, I can.", "It is."), so they are not listed here.
_NEVER_FINAL = frozenset(
    "and but or so to of in on at with for from the a an um uh like because "
    "if when while my your our their his her its "
    "i'm it's we're they're you're i've we've i'd we'd that's there's he's she's".split()
)


def _last_word(text: str) -> str:
    """Last token with surrounding punctuation stripped, lower-cased."""
    words = text.rstrip().rstrip("".join(_TERMINAL)).split()
    return words[-1].lower().strip(_EDGE_PUNCT) if words else ""


def looks_complete(text: str) -> bool:
    """Terminal punctuation AND a last word that can end a thought.

    ASR punctuates confidently even when it guesses: on narrowband (phone)
    audio Deepgram produced "Yeah. Sure. I mean, if." and the fast tier cut
    the answer in half (found by the Twilio-protocol simulator). A period after
    a conjunction, preposition, article, filler or possessive is a mid-thought
    pause, so it is handed to the trailing tier instead. Modals and copulas
    are NOT overridden: "Yes, I can." is a complete answer."""
    text = text.rstrip()
    if not text.endswith(_TERMINAL):
        return False
    return _last_word(text) not in _NEVER_FINAL


def looks_trailing(text: str) -> bool:
    text = text.rstrip()
    if text.endswith((",", ";", ":")):
        return True  # mid-clause punctuation: the strongest "still going" cue
    last = _last_word(text)
    return bool(last) and (last in _TRAILING_WORDS or last in _TRAILING_CONTRACTIONS)


@dataclass(frozen=True)
class TurnComplete:
    transcript: str
    endpoint_delay: float  # seconds from vad_stop to commit
    vad_stop_t: float
    commit_t: float
    reason: str  # "fast" | "slow" | "trailing" | "utterance_end"


class Endpointer:
    def __init__(self, fast_sec: float = 0.20, slow_sec: float = 2.00,
                 trailing_sec: float = 2.50) -> None:
        self.fast_sec = fast_sec
        self.slow_sec = slow_sec
        self.trailing_sec = trailing_sec
        self._finals: list[str] = []
        self._vad_stop_t: float | None = None  # None = speaking or idle, nothing armed

    @property
    def transcript(self) -> str:
        return " ".join(self._finals)

    @property
    def armed(self) -> bool:
        """Caller is silent and a commit deadline is running."""
        return self._vad_stop_t is not None

    @property
    def pending_complete(self) -> bool:
        """Armed (caller silent) with a complete-looking transcript: the fast
        tier will commit at its deadline. Downstream may start SPECULATIVE
        generation now — audio release still waits for the actual commit."""
        return (self._vad_stop_t is not None and bool(self.transcript)
                and looks_complete(self.transcript))

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
