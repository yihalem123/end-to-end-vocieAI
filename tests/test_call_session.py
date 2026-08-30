"""Consent-gated caller turns and stable transcript identities."""
import asyncio
from pathlib import Path

from server.engine.plan import InterviewState, load_plan
from server.realtime.call import CallSession, CallState
from server.realtime.session import SessionLifecycle, SessionStatus

PLAN_PATH = Path(__file__).resolve().parents[1] / "plans" / "icu_nurse.yaml"


class FakeReplies:
    def __init__(self) -> None:
        self.interview = InterviewState(load_plan(PLAN_PATH))
        self.scripts: list[tuple[str, int]] = []
        self.chats: list[tuple[str, int]] = []

    async def on_script(self, text: str, turn_id: int) -> None:
        self.scripts.append((text, turn_id))

    async def on_chat(self, text: str, turn_id: int) -> None:
        self.chats.append((text, turn_id))


def make_session() -> tuple[CallSession, FakeReplies, list[dict]]:
    session = object.__new__(CallSession)
    lifecycle = SessionLifecycle("call-abc")
    lifecycle.transition(SessionStatus.AWAITING_CONSENT)
    session.state = CallState(call_id="call-abc", session=lifecycle)
    replies = FakeReplies()
    session._replies = replies
    session._close_when_idle = False
    sent: list[dict] = []

    async def send(payload: dict) -> None:
        sent.append(payload)

    session._send = send
    return session, replies, sent


def test_explicit_consent_enters_interview_with_stable_utterance_id() -> None:
    async def run() -> None:
        session, replies, sent = make_session()
        await session._commit_caller_text("Yes, I consent.", 1, turn=None)
        assert session.state.session.status == SessionStatus.INTERVIEWING
        assert session.state.conversation == [{
            "role": "caller", "text": "Yes, I consent.", "turn_id": 1,
            "utterance_id": "call-abc:u1", "call_id": "call-abc",
        }]
        assert replies.interview.fields["consent"].value is True
        assert replies.chats == [("Yes, I consent.", 1)]
        assert sent[0]["utterance_id"] == "call-abc:u1"

    asyncio.run(run())


def test_consent_refusal_stops_before_engine_processing() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        await session._commit_caller_text("No, I do not consent.", 1, turn=None)
        assert session.state.session.status == SessionStatus.CONSENT_REFUSED
        assert replies.chats == []
        assert replies.scripts[-1][0].startswith("Understood")
        assert session._close_when_idle is True

    asyncio.run(run())


def test_ambiguous_consent_reprompts_without_engine_or_false_value() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        await session._commit_caller_text("Maybe, I'm not sure.", 1, turn=None)
        assert session.state.session.status == SessionStatus.AWAITING_CONSENT
        assert "consent" not in replies.interview.fields
        assert replies.chats == []
        assert "clear yes or no" in replies.scripts[-1][0]

    asyncio.run(run())


def test_explicit_end_request_bypasses_llm_and_closes_gracefully() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        session.state.session.transition(SessionStatus.INTERVIEWING)
        await session._commit_caller_text("I want you to end this call.", 2, turn=None)
        assert session.state.session.status == SessionStatus.CLOSING
        assert replies.chats == []
        assert replies.interview.end_call_request.reason == "candidate_requested"
        assert "end the call" in replies.scripts[-1][0]
        assert session._close_when_idle is True

    asyncio.run(run())


def test_fuzzy_end_request_asks_for_confirmation_not_an_interview_answer() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        session.state.session.transition(SessionStatus.INTERVIEWING)
        await session._commit_caller_text(
            "Can you in the cold in the cold?", 2, turn=None)
        assert session.state.session.status == SessionStatus.INTERVIEWING
        assert replies.chats == []
        assert "ask me to end" in replies.scripts[-1][0]
        assert session._pending_end_confirmation is True

        await session._commit_caller_text("Yes.", 3, turn=None)
        assert session.state.session.status == SessionStatus.CLOSING
        assert replies.interview.end_call_request.reason == "candidate_requested"

    asyncio.run(run())


def test_repeated_fuzzy_end_request_confirms_without_waiting_for_exact_yes() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        session.state.session.transition(SessionStatus.INTERVIEWING)
        await session._commit_caller_text("Please in the call.", 2, turn=None)
        assert session._pending_end_confirmation is True

        await session._commit_caller_text("in this call", 3, turn=None)
        assert session.state.session.status == SessionStatus.CLOSING
        assert replies.interview.end_call_request.reason == "candidate_requested"
        assert replies.chats == []

    asyncio.run(run())


def test_event_loop_agent_initiates_with_disclosure() -> None:
    async def run() -> None:
        session, replies, _ = make_session()
        session.state.session = SessionLifecycle("call-abc")
        session._events = asyncio.Queue()
        session.mode = "flux"
        initiated = asyncio.Event()

        async def on_script(text: str, turn_id: int) -> None:
            replies.scripts.append((text, turn_id))
            initiated.set()

        replies.on_script = on_script
        task = asyncio.create_task(session._event_loop())
        await asyncio.wait_for(initiated.wait(), timeout=0.2)
        assert session.state.session.status == SessionStatus.AWAITING_CONSENT
        assert "transcribe" in replies.scripts[0][0]
        assert replies.scripts[0][1] == 0
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def test_stale_asr_epoch_is_rejected_and_counted() -> None:
    session = object.__new__(CallSession)
    session.state = CallState(
        call_id="call-abc", session=SessionLifecycle("call-abc"))
    session._asr_epoch = 3
    assert session._accept_asr_epoch(3)
    assert not session._accept_asr_epoch(2)
    assert session.state.stale_asr_events == 1
