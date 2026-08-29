"""Barge-in onset guard: sustained speech during agent audio triggers a cut."""
from server.realtime.bargein import BargeInGuard

GUARD = 0.25


def make() -> BargeInGuard:
    return BargeInGuard(onset_sec=GUARD)


def test_speech_while_agent_silent_never_triggers() -> None:
    g = make()
    g.on_vad_start(t=1.0)
    assert g.tick(t=5.0) is False        # normal turn-taking, not barge-in


def test_sustained_speech_during_agent_audio_triggers() -> None:
    g = make()
    g.on_agent_audio_start()
    g.on_vad_start(t=1.0)
    assert g.tick(t=1.0 + GUARD - 0.05) is False   # not sustained yet
    assert g.tick(t=1.0 + GUARD + 0.01) is True    # sustained -> cut the agent


def test_short_blip_does_not_trigger() -> None:
    g = make()
    g.on_agent_audio_start()
    g.on_vad_start(t=1.0)
    g.on_vad_stop(t=1.15)                # cough/noise: shorter than the guard
    assert g.tick(t=2.0) is False


def test_agent_finishing_disarms_pending_guard() -> None:
    # Agent audio ends while the guard is arming: the user isn't interrupting
    # anymore, they're just taking their turn.
    g = make()
    g.on_agent_audio_start()
    g.on_vad_start(t=1.0)
    g.on_agent_audio_end()
    assert g.tick(t=1.0 + GUARD + 0.5) is False


def test_triggers_once_per_onset() -> None:
    g = make()
    g.on_agent_audio_start()
    g.on_vad_start(t=1.0)
    assert g.tick(t=1.3) is True
    assert g.tick(t=1.4) is False        # already cut; don't re-fire


def test_new_agent_audio_re_arms_for_next_interruption() -> None:
    g = make()
    g.on_agent_audio_start()
    g.on_vad_start(t=1.0)
    assert g.tick(t=1.3) is True         # first barge-in
    g.on_vad_stop(t=1.8)
    g.on_agent_audio_start()             # agent answers again
    g.on_vad_start(t=3.0)
    assert g.tick(t=3.3) is True         # second barge-in works the same way
