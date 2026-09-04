"""The evidence gate: validate, apply and ledger every tool call.

## How this works
The LLM is a witness, not a judge. apply_tools() is the only path by which a
model output mutates interview state, and every call - applied or not - is
appended to the tool ledger with an idempotency key (call scope + turn + tool
identity) so a replayed generation is skipped rather than double-recorded.
record_answer must carry a verbatim quote that lexically anchors to SOME
committed caller utterance (any turn, not just the current one: the model
legitimately records volunteered answers a turn late). end_call is checked
against the transcript, the coverage state and the runtime budget; the
backend owns the lifecycle transition either way. Evidence mutations settle
before end_call is validated regardless of the model's output order.
"""
import hashlib
import json
import logging
import re
from collections.abc import Callable

from server.engine.intents import EndCallIntent, classify_end_call_intent
from server.engine.plan import InterviewState
from server.engine.stream import ToolCall

log = logging.getLogger(__name__)


def apply_tools(
    state: InterviewState,
    calls: list[ToolCall],
    *,
    call_id: str,
    is_current: Callable[[], bool] | None = None,
    turn_id: int | None = None,
    generation_id: int | None = None,
    source_text: str = "",
) -> list[dict]:
    """Validate, apply, and LEDGER every tool call. Failures are returned to
    orchestration as ledger entries (and warnings), never swallowed; duplicate
    tool_call_ids are idempotently skipped."""
    current = is_current or (lambda: True)
    ledger = state.tool_ledger
    seen = {e["idempotency_key"] for e in ledger
            if e.get("idempotency_key")}
    results: list[dict] = []
    # Function-call output order is model-controlled. Evidence mutations
    # must settle before an interview_complete request is validated, or an
    # end_call item emitted first can be rejected even though the same
    # response records the final objective. Stable sort preserves relative
    # order within each class and keeps the audit deterministic.
    ordered_calls = sorted(calls, key=lambda call: call.name == "end_call")
    for call in ordered_calls:
        if not current():
            return results
        tool_identity = call.call_id or hashlib.sha256(
            (call.name + json.dumps(call.arguments, sort_keys=True,
                                    default=str)).encode("utf-8")
        ).hexdigest()[:16]
        idempotency_key = f"{call_id}:{turn_id or 0}:{tool_identity}"
        execution_id = (f"{call_id}:{turn_id or 0}:"
                        f"{generation_id or 0}:{tool_identity}")
        entry = {"call_id": call_id,
                 "tool_call_id": call.call_id, "name": call.name,
                  "turn_id": turn_id, "generation_id": generation_id,
                 "execution_id": execution_id,
                 "idempotency_key": idempotency_key,
                  "arguments": call.arguments, "applied": False, "reason": ""}
        ledger.append(entry)
        results.append(entry)
        if idempotency_key in seen:
            entry["reason"] = "duplicate idempotency identity (skip)"
            continue
        seen.add(idempotency_key)
        if call.arguments is None:
            entry["reason"] = "malformed arguments"
            log.warning("malformed tool args for %s; skipped", call.name)
            continue
        if call.name == "record_answer":
            quote = str(call.arguments.get("quote") or "").strip()
            if not quote:
                entry["reason"] = "empty quote: evidence required before mutation"
                log.warning("record_answer without evidence rejected")
                continue
            if not any(quote_supported(quote, s) for s in
                       caller_sources(state, source_text)):
                entry["reason"] = "quote not supported by any caller utterance"
                log.warning("record_answer with unsupported evidence rejected")
                continue
            entry["applied"] = state.record(
                call.arguments.get("field", ""),
                call.arguments.get("value"), quote)
            if not entry["applied"]:
                entry["reason"] = "rejected by state validation"
                log.warning("record_answer rejected by state validation")
        elif call.name == "end_call":
            reason = str(call.arguments.get("reason") or "")
            closing = str(call.arguments.get("closing_message") or "")
            if (reason == "candidate_requested"
                    and not any(classify_end_call_intent(source)
                                == EndCallIntent.END
                                for source in caller_sources(state, source_text))):
                entry["reason"] = "candidate end request not supported by transcript"
                continue
            if reason == "interview_complete" and state.remaining:
                entry["reason"] = "interview objectives still missing"
                continue
            if (reason == "limit_reached"
                    and state.caller_turn_count <
                    state.plan.boundaries.max_turns
                    and state.elapsed_seconds <
                    state.plan.boundaries.max_duration_minutes * 60):
                entry["reason"] = "interview limit not reached"
                continue
            entry["applied"] = state.request_end_call(reason, closing)
            if not entry["applied"]:
                entry["reason"] = "invalid or duplicate end-call request"
        else:
            entry["reason"] = "unknown tool"
    return results


def caller_sources(state: InterviewState, source_text: str) -> list[str]:
    """Evidence may come from ANY committed caller utterance, not just the
    current one — the model legitimately records volunteered or delayed
    answers a turn late (live finding: single-utterance scoping caused
    systematic rejects and re-ask loops). History holds only committed,
    ownership-gated user turns, so this stays anchored to real speech."""
    sources = [h["content"] for h in state.history
               if h.get("role") == "user"]
    sources.append(source_text)
    return [s for s in sources if s]


def quote_supported(quote: str, source_text: str) -> bool:
    """Loose lexical anchoring: punctuation/case may differ, words may not."""

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    evidence = normalize(quote)
    source = normalize(source_text)
    return bool(evidence) and evidence in source
