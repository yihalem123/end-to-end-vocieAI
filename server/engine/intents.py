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
        r"\b(?:do not|don t|no longer) want to continue (?:with )?"
        r"(?:this |the )?(?:call|interview|screening)\b",
        r"\bi want (?:you )?to (?:end|stop) (?:this |the )?"
        r"(?:call|interview|screening)\b",
        r"\bplease (?:end|stop) (?:this |the )?"
        r"(?:call|interview|screening)\b",
    )
    if normalized == "stop" or any(re.search(pattern, normalized) for pattern in direct):
        return EndCallIntent.END

    # Without an explicit call/interview object these phrases are destructive
    # ambiguities: "stop working nights" and "continue night shifts" are
    # ordinary screening answers. Only a complete bare request is confirmable.
    ambiguous_request = (
        r"^(?:i )?(?:do not|don t|no longer) want to continue$",
        r"^i want (?:you )?to (?:end|stop)$",
        r"^please (?:end|stop)$",
    )
    if any(re.fullmatch(pattern, normalized) for pattern in ambiguous_request):
        return EndCallIntent.CONFIRM

    # Narrow phonetic neighborhoods observed in live calls. These only request
    # confirmation; they never terminate by themselves.
    fuzzy_phrase = re.search(
        r"\b(?:in|into|end) (?:this |the )?(?:call|cold)\b", normalized)
    request_cue = ({"please", "can", "could", "want", "end", "stop"}
                   & set(normalized.split()))
    if fuzzy_phrase and (request_cue or normalized in {
            "in this call", "in the call", "into this call"}):
        return EndCallIntent.CONFIRM
    return None
