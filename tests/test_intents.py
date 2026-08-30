"""High-authority call termination intent and narrow ASR repair."""
import pytest

from server.engine.intents import EndCallIntent, classify_end_call_intent


@pytest.mark.parametrize("text", [
    "I want you to end this call.",
    "Please hang up.",
    "Stop the interview.",
    "I do not want to continue.",
    "stop",
])
def test_explicit_end_call_intent(text: str) -> None:
    assert classify_end_call_intent(text) == EndCallIntent.END


@pytest.mark.parametrize("text", [
    "Can you in the cold in the cold?",
    "Enough enough honey in the cold, please.",
    "Please in the call.",
    "in this call",
    "into this call",
])
def test_observed_asr_confusion_requires_confirmation(text: str) -> None:
    assert classify_end_call_intent(text) == EndCallIntent.CONFIRM


@pytest.mark.parametrize("text", [
    "It is cold in Delaware.",
    "I worked in the cold storage unit.",
    "Please tell me about the role.",
])
def test_unrelated_language_does_not_end_call(text: str) -> None:
    assert classify_end_call_intent(text) is None
