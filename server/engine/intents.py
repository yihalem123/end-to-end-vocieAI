"""Small control-intent classifiers shared by orchestration and tool validation."""
import re
from enum import StrEnum


class EndCallIntent(StrEnum):
    END = "end"
    CONFIRM = "confirm"


def classify_end_call_intent(text: str) -> EndCallIntent | None:
    """Detect explicit termination and a narrow, observed ASR confusion.

    Fuzzy matches never terminate a call: they only ask for confirmation. This
    catches Deepgram's observed "end the call" -> "in the cold" substitution
    without treating arbitrary mentions of cold weather as a hangup command.
    """
    normalized = " ".join(re.findall(r"[a-z]+", text.casefold()))
    if not normalized:
        return None
    direct = (
        r"\b(?:end|stop) (?:this |the )?(?:call|interview|screening)\b",
        r"\bhang up\b",
        r"\b(?:do not|don t|no longer) want to continue\b",
        r"\bi want (?:you )?to (?:end|stop)\b",
        r"\bplease (?:end|stop)\b",
    )
    if normalized == "stop" or any(re.search(pattern, normalized) for pattern in direct):
        return EndCallIntent.END

    # Only the exact, repeatedly observed phonetic neighborhood plus request
    # language qualifies. The caller must confirm before any state transition.
    fuzzy_phrase = re.search(r"\b(?:in|end) the cold\b", normalized)
    request_cue = ({"please", "can", "could", "want", "end", "stop"}
                   & set(normalized.split()))
    if fuzzy_phrase and request_cue:
        return EndCallIntent.CONFIRM
    return None
