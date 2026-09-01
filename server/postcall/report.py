"""Post-call pipeline trigger + in-memory report store. Phase 5b.

## How this works
After a consented call ends, app.py fires run_postcall() under the call id:
extract caller-id evidence -> verify -> score deterministically -> store. Refused
or cancelled sessions get terminal, unscored reports without extraction. The
store is a bounded in-process dict (oldest-first eviction) — a
database is a production talking point, not a demo requirement. build_report()
is pure so the report shape is unit-testable without the vendor call; the HTML
view lives in report_view.py (the Call Report design canvas).
"""
import logging
import time
from collections import OrderedDict

from server.config import Settings
from server.engine.plan import load_plan_cached
from server.metrics import registry
from server.postcall.analyze import analyze_call
from server.postcall.extract import Extracted, extract_call
from server.postcall.report_view import render_report_html
from server.postcall.score import ScoreResult, score_call
from server.realtime.session import SessionLifecycle, SessionStatus

log = logging.getLogger(__name__)

MAX_REPORTS = 50
reports: OrderedDict[str, dict] = OrderedDict()


def build_report(call_id: str, conversation: list[dict], turns: list,
                 extracted: dict[str, Extracted], result: ScoreResult,
                 session_state: str = SessionStatus.COMPLETED,
                 tool_ledger: list[dict] | None = None,
                 analyses: list[dict] | None = None,
                 latency: dict | None = None) -> dict:
    return {
        "analyses": list(analyses or []),
        "call_id": call_id,
        "session_state": session_state,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": result.score,
        "needs_review": result.needs_review,
        "knocked_out": result.knocked_out,
        "scoring_version": result.scoring_version,
        "reasons": result.reasons,
        "subscores": result.subscores,
        "fields": {
            name: {"value": e.value, "quote": e.quote, "confidence": e.confidence,
                   "utterance_id": e.utterance_id, "verified": e.verified,
                   "contradictory": e.contradictory}
            for name, e in extracted.items()
        },
        "tool_ledger": list(tool_ledger or []),
        "turn_count": len(turns),
        "endpoint_delays_ms": [round(t.endpoint_delay * 1000) for t in turns],
        # Every measured stage, so the report can decompose turn latency
        # instead of showing only where the endpointer spent its time.
        "latency": dict(latency or {}),
        "conversation": conversation,
    }


def _store(call_id: str, report: dict) -> None:
    reports[call_id] = report
    while len(reports) > MAX_REPORTS:
        reports.popitem(last=False)


def store_terminal_report(call_id: str, conversation: list[dict],
                          status: SessionStatus, reason: str) -> None:
    _store(call_id, {
        "call_id": call_id,
        "session_state": status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": None,
        "needs_review": status not in {SessionStatus.CONSENT_REFUSED,
                                        SessionStatus.CANCELLED},
        "knocked_out": None,
        "scoring_version": None,
        "reasons": [reason],
        "subscores": {}, "fields": {}, "turn_count": 0,
        "endpoint_delays_ms": [], "conversation": conversation,
    })


async def run_postcall(call_id: str, conversation: list[dict], turns: list,
                       settings: Settings, lifecycle: SessionLifecycle | None = None,
                       tool_ledger: list[dict] | None = None) -> None:
    try:
        plan = load_plan_cached(str(settings.plan_path))

        extracted = await extract_call(settings, plan, conversation)
        result = score_call(plan, extracted)
        report = build_report(
            call_id, conversation, turns, extracted, result,
            session_state=SessionStatus.COMPLETED, tool_ledger=tool_ledger,
            analyses=[], latency=registry.snapshot(call_id)["stages"])
        if lifecycle is not None:
            lifecycle.transition(SessionStatus.COMPLETED)
        _store(call_id, report)
        # The evidence report is available now. Advisory generation may be slow
        # or fail independently; it enriches the stored report when ready.
        if not settings.openai_api_key:
            return
        try:
            analyses = await analyze_call(settings, plan, conversation)
        except Exception:
            log.exception("advisory analyses failed for %s", call_id)
        else:
            report["analyses"] = analyses
        log.info("postcall %s: score=%s needs_review=%s knocked_out=%s",
                 call_id, result.score, result.needs_review, result.knocked_out)
    except Exception:
        log.exception("postcall pipeline failed for %s", call_id)
        if lifecycle is not None and lifecycle.status == SessionStatus.POST_PROCESSING:
            lifecycle.transition(SessionStatus.FAILED)
        store_terminal_report(
            call_id, conversation, SessionStatus.FAILED,
            "postcall failed — see logs")


def render_html(report: dict) -> str:
    return render_report_html(report)
