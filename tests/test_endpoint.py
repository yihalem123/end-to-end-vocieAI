"""Endpointer state machine: fast/slow timers, UtteranceEnd, cancellation."""
from server.realtime.endpoint import Endpointer

FAST = 0.25
SLOW = 1.20


def make() -> Endpointer:
    return Endpointer(fast_sec=FAST, slow_sec=SLOW)


def test_fast_commit_for_complete_sentence() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("I have five years of ICU experience.", t=1.0)
    ep.on_vad_stop(t=1.1)
    assert ep.tick(t=1.1 + FAST - 0.05) is None          # before the fast deadline
    turn = ep.tick(t=1.1 + FAST + 0.01)
    assert turn is not None
    assert turn.transcript == "I have five years of ICU experience."
    assert turn.reason == "fast"
    assert abs(turn.endpoint_delay - (FAST + 0.01)) < 1e-9


def test_incomplete_text_waits_for_slow_timer() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("I worked at", t=1.0)                # trailing off, no punctuation
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + FAST + 0.1) is None           # fast timer must NOT fire
    turn = ep.tick(t=1.0 + SLOW + 0.01)
    assert turn is not None
    assert turn.reason == "slow"


def test_vad_start_cancels_pending_commit() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("I worked at Kaiser.", t=1.0)
    ep.on_vad_stop(t=1.0)
    ep.on_vad_start(t=1.1)                               # speaker resumes mid-pause
    assert ep.tick(t=5.0) is None                        # nothing may commit now
    ep.on_asr_final("And then at Stanford.", t=5.5)
    ep.on_vad_stop(t=5.5)
    turn = ep.tick(t=5.5 + FAST + 0.01)
    assert turn is not None
    assert turn.transcript == "I worked at Kaiser. And then at Stanford."


def test_utterance_end_commits_immediately() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("Yes", t=1.0)                        # no punctuation -> slow path
    ep.on_vad_stop(t=1.0)
    turn = ep.on_utterance_end(t=1.4)                    # Deepgram is confident: commit
    assert turn is not None
    assert turn.reason == "utterance_end"
    assert abs(turn.endpoint_delay - 0.4) < 1e-9


def test_late_final_upgrades_slow_to_fast() -> None:
    # ASR lags speech: at vad_stop no text exists yet, so the slow timer arms.
    # When the punctuated final arrives, the deadline must re-arm to fast,
    # measured from vad_stop (not from the final's arrival).
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.1) is None
    ep.on_asr_final("Two years in the ER.", t=1.15)
    turn = ep.tick(t=1.0 + FAST + 0.01)
    assert turn is not None
    assert turn.reason == "fast"


def test_no_text_never_commits() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_vad_stop(t=1.0)                                # cough: VAD fired, ASR heard nothing
    assert ep.tick(t=1.0 + SLOW + 5.0) is None
    assert ep.on_utterance_end(t=7.0) is None


def test_state_resets_between_turns() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("First answer.", t=1.0)
    ep.on_vad_stop(t=1.0)
    first = ep.tick(t=1.0 + FAST + 0.01)
    assert first is not None and first.transcript == "First answer."
    ep.on_vad_start(t=2.0)
    ep.on_asr_final("Second answer.", t=3.0)
    ep.on_vad_stop(t=3.0)
    second = ep.tick(t=3.0 + FAST + 0.01)
    assert second is not None
    assert second.transcript == "Second answer."         # no bleed from turn one
