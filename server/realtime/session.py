"""Explicit per-call candidate-session lifecycle and consent classification."""
import re
from dataclasses import dataclass, field
from enum import StrEnum


class SessionStatus(StrEnum):
    DISCLOSURE = "disclosure"
    AWAITING_CONSENT = "awaiting_consent"
    INTERVIEWING = "interviewing"
    CLOSING = "closing"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    CONSENT_REFUSED = "consent_refused"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED = {
    SessionStatus.DISCLOSURE: {
        SessionStatus.AWAITING_CONSENT, SessionStatus.FAILED, SessionStatus.CANCELLED},
    SessionStatus.AWAITING_CONSENT: {
        SessionStatus.INTERVIEWING, SessionStatus.CONSENT_REFUSED,
        SessionStatus.FAILED, SessionStatus.CANCELLED},
    SessionStatus.INTERVIEWING: {
        SessionStatus.CLOSING, SessionStatus.POST_PROCESSING,
        SessionStatus.FAILED, SessionStatus.CANCELLED},
    SessionStatus.CLOSING: {
        SessionStatus.POST_PROCESSING, SessionStatus.COMPLETED,
        SessionStatus.FAILED, SessionStatus.CANCELLED},
    SessionStatus.POST_PROCESSING: {
        SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED},
}


@dataclass
class SessionLifecycle:
    call_id: str
    status: SessionStatus = SessionStatus.DISCLOSURE
    history: list[SessionStatus] = field(
        default_factory=lambda: [SessionStatus.DISCLOSURE])

    def transition(self, target: SessionStatus) -> None:
        if target == self.status:
            return
        if target not in _ALLOWED.get(self.status, set()):
            raise ValueError(f"invalid session transition {self.status} -> {target}")
        self.status = target
        self.history.append(target)


def classify_consent(text: str) -> bool | None:
    """Conservative yes/no classifier. Ambiguity stays nullable."""
    normalized = " ".join(re.findall(r"[a-z]+", text.casefold()))
    words = set(normalized.split())
    if not normalized or {"unsure", "maybe"} & words or "not sure" in normalized:
        return None
    if ({"no", "decline", "refuse", "stop"} & words or "do not" in normalized
            or "don t" in normalized or "not now" in normalized):
        return False
    if ({"yes", "consent", "agree", "sure", "okay", "ok"} & words
            or "go ahead" in normalized):
        return True
    return None
