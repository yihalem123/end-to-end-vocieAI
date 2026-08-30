"""Endpointer state machine: patience tiers, UtteranceEnd, cancellation."""
from server.realtime.endpoint import Endpointer

FAST = 0.25
SLOW = 2.00
TRAILING = 2.50


def make() -> Endpointer:
    return Endpointer(fast_sec=FAST, slow_sec=SLOW, trailing_sec=TRAILING)


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
    ep.on_asr_final("I've been a nurse for three", t=1.0)  # unpunctuated, mid-list
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + FAST + 0.1) is None           # fast timer must NOT fire
    turn = ep.tick(t=1.0 + SLOW + 0.01)
    assert turn is not None
    assert turn.reason == "slow"


def test_trailing_word_gets_extra_patience() -> None:
    # Live regression (2026-08-30): "Since I was telling you that, it's really
    # great to" was cut at the slow timer. A dangling function word means the
    # thought is still in flight — longest patience tier.
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("Since I was telling you that, it's really great to", t=1.0)
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + SLOW + 0.1) is None           # slow tier must NOT fire
    turn = ep.tick(t=1.0 + TRAILING + 0.01)
    assert turn is not None
    assert turn.reason == "trailing"


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


def test_utterance_end_commits_complete_text_immediately() -> None:
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("Yes, I can.", t=1.0)
    ep.on_vad_stop(t=1.0)
    turn = ep.on_utterance_end(t=1.4)                    # word gap + complete: commit
    assert turn is not None
    assert turn.reason == "utterance_end"
    assert abs(turn.endpoint_delay - 0.4) < 1e-9


def test_trailing_copula_gets_extra_patience() -> None:
    # Simulated-caller regression: "So my license is" / "Honestly I prefer"
    # split at the slow tier when the pause ran long. A trailing copula or
    # auxiliary is a thought in flight.
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("So my license is", t=1.0)
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + SLOW + 0.1) is None
    turn = ep.tick(t=1.0 + TRAILING + 0.01)
    assert turn is not None and turn.reason == "trailing"


def test_dangling_contraction_gets_extra_patience() -> None:
    # Live regression: "I guess it's" hit the 2 s slow timeout, became its own
    # caller turn, and made the agent repeat the question over the continuation.
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("I guess it's", t=1.0)
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + SLOW + 0.1) is None
    turn = ep.tick(t=1.0 + TRAILING + 0.01)
    assert turn is not None and turn.reason == "trailing"


def test_trailing_comma_gets_extra_patience() -> None:
    # Live regression (2026-08-30 #2): "Yeah. I mean," was committed at the
    # slow tier — but a trailing comma is the strongest "still going" cue.
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("Yeah. I mean,", t=1.0)
    ep.on_vad_stop(t=1.0)
    assert ep.tick(t=1.0 + SLOW + 0.1) is None
    turn = ep.tick(t=1.0 + TRAILING + 0.01)
    assert turn is not None
    assert turn.reason == "trailing"


def test_utterance_end_defers_to_timers_on_incomplete_text() -> None:
    # Live regression (2026-08-30): "But I'm not I'm not sure, though. And" was
    # committed 157 ms after vad_stop because UtteranceEnd bypassed the
    # completeness check. UtteranceEnd is evidence, not a command.
    ep = make()
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("But I'm not I'm not sure, though. And", t=1.0)
    ep.on_vad_stop(t=1.0)
    assert ep.on_utterance_end(t=1.16) is None           # trailing "And": wait
    assert ep.tick(t=1.0 + SLOW + 0.1) is None
    turn = ep.tick(t=1.0 + TRAILING + 0.01)              # patience, then commit
    assert turn is not None
    assert turn.reason == "trailing"


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


def test_pending_complete_gates_speculation() -> None:
    ep = make()
    assert ep.pending_complete is False
    ep.on_vad_start(t=0.0)
    ep.on_asr_final("I have five years.", t=1.0)
    assert ep.pending_complete is False        # still speaking: not armed
    ep.on_vad_stop(t=1.1)
    assert ep.pending_complete is True         # armed + complete: speculate
    ep.on_asr_final("And also", t=1.2)
    assert ep.pending_complete is False        # transcript no longer complete
