"""High-authority call termination intent and narrow ASR repair."""
import pytest

from server.engine.intents import EndCallIntent, classify_end_call_intent


@pytest.mark.parametrize("text", [
    "I want you to end this call.",
    "Please hang up.",
    "Stop the interview.",
    "Bye.",
    "Okay, goodbye.",
    "And not sure. Bye.",
    "I do not want to continue this interview.",
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
    "I do not want to continue.",
    "I want to stop.",
    "Please stop.",
])
def test_observed_asr_confusion_requires_confirmation(text: str) -> None:
    assert classify_end_call_intent(text) == EndCallIntent.CONFIRM


@pytest.mark.parametrize("text", [
    "It is cold in Delaware.",
    "I worked in the cold storage unit.",
    "Please tell me about the role.",
    "Bye the way, I have a question.",
    "I want to stop working nights.",
    "I want to stop doing night shifts.",
    "I do not want to continue night shifts.",
    "Please stop by the unit if you can.",
    "Please stop me if I am wrong.",
])
def test_unrelated_language_does_not_end_call(text: str) -> None:
    assert classify_end_call_intent(text) is None
