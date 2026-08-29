"""Candidate session lifecycle and conservative consent classification."""
import pytest

from server.realtime.session import SessionLifecycle, SessionStatus, classify_consent


def test_happy_path_session_states_are_explicit() -> None:
    lifecycle = SessionLifecycle("call-1")
    for status in (
        SessionStatus.AWAITING_CONSENT,
        SessionStatus.INTERVIEWING,
        SessionStatus.CLOSING,
        SessionStatus.POST_PROCESSING,
        SessionStatus.COMPLETED,
    ):
        lifecycle.transition(status)
    assert lifecycle.history == [
        SessionStatus.DISCLOSURE, SessionStatus.AWAITING_CONSENT,
        SessionStatus.INTERVIEWING, SessionStatus.CLOSING,
        SessionStatus.POST_PROCESSING, SessionStatus.COMPLETED,
    ]


def test_consent_refusal_is_terminal() -> None:
    lifecycle = SessionLifecycle("call-2")
    lifecycle.transition(SessionStatus.AWAITING_CONSENT)
    lifecycle.transition(SessionStatus.CONSENT_REFUSED)
    with pytest.raises(ValueError, match="invalid session transition"):
        lifecycle.transition(SessionStatus.INTERVIEWING)


@pytest.mark.parametrize("text", ["Yes", "sure, go ahead", "I consent", "okay"])
def test_explicit_consent_is_accepted(text: str) -> None:
    assert classify_consent(text) is True


@pytest.mark.parametrize("text", ["No", "I do not consent", "not now", "stop"])
def test_explicit_refusal_is_rejected(text: str) -> None:
    assert classify_consent(text) is False


@pytest.mark.parametrize("text", ["maybe", "I'm not sure", "hello", ""])
def test_ambiguous_consent_stays_unknown(text: str) -> None:
    assert classify_consent(text) is None
