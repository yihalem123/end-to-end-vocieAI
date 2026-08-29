"""Scripted stand-in for the interview engine. Phase 3 only.

## How this works
The Phase 3 voice loop needs SOMETHING to say so TTS, pacing, and barge-in can
be built and profiled without the LLM. This cycles through a canned ICU-screen
script, one reply per committed caller turn. Replies are pre-split into
sentences because the speak task opens one TTS stream per sentence and records
a frame mark at each boundary — that's what makes barge-in truncation exact.
Phase 4 replaces this object with the real engine behind the same call shape:
reply(transcript) -> list of sentences.
"""

_SCRIPT: list[list[str]] = [
    ["Thanks for calling about the ICU position.",
     "To start, can you tell me about your current role?"],
    ["Got it, thank you.",
     "How many years of ICU experience do you have?"],
    ["That's helpful.",
     "Are you comfortable working night shifts and weekends?"],
    ["Understood.",
     "Do you hold an active RN license in this state?"],
    ["Great, that's everything I needed for this screen.",
     "The recruiting team will follow up with next steps soon."],
]
_FALLBACK = ["Could you say a bit more about that?"]


class StubEngine:
    def __init__(self) -> None:
        self._idx = 0

    def reply(self, transcript: str) -> list[str]:
        if self._idx < len(_SCRIPT):
            sentences = _SCRIPT[self._idx]
            self._idx += 1
            return sentences
        return list(_FALLBACK)
