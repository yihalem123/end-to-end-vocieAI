"""Prompt assembly: what the model is told, and in what order.

## How this works
Two requests per caller turn share one plan but read differently:
- build_instructions(): the STATIC caller-facing prefix (persona, evidence
  objectives, boundaries, rules). Byte-identical across turns.
- build_extraction_instructions(): the static prefix for the non-speaking
  evidence pass that may call the tools in TOOLS.
- build_state_block(): the VOLATILE per-turn coverage state - recorded fields
  (quotes truncated so the prompt never grows with caller verbosity), missing
  objectives, runtime budget, and the current objective line.
cacheable_input() orders the request static -> stable history -> volatile
state -> newest turn. That is the cache-friendly layout, though it does not
buy prompt caching here (its docstring records the measurements). Plan data
stays in plans/*.yaml; nothing here hard-codes a question.
"""
from server.engine.plan import InterviewState

TOOLS = [
    {
        "type": "function",
        "name": "record_answer",
        "description": "Record one answered field with the caller's verbatim words.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "value", "quote"],
            "properties": {
                "field": {"type": "string", "description": "Plan field name."},
                "value": {"type": "string",
                          "description": "The answer; lists comma-separated, booleans true/false."},
                "quote": {"type": "string",
                          "description": "Verbatim words the caller said."},
            },
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": (
            "Request a graceful end after the caller asks to stop or all "
            "interview objectives are covered. The backend validates and owns "
            "the actual session transition."),
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reason", "closing_message"],
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": [
                        "candidate_requested", "interview_complete",
                        "limit_reached",
                    ],
                },
                "closing_message": {
                    "type": "string",
                    "description": (
                        "One short final statement, with no question, to speak "
                        "before the backend closes the call."),
                },
            },
        },
    },
]


def fallback_line(state: InterviewState) -> str:
    """Last-resort speech if the dialogue request produces no text."""
    if state.end_call_request is not None:
        return state.end_call_request.closing_message
    if state.remaining:
        return "I'm sorry, I lost my place for a moment. What would you like me to clarify?"
    return "Thanks, that covers everything I needed for this screening."


def build_instructions(plan) -> str:
    """The STATIC per-call prefix (persona + rules). Kept byte-identical across
    turns so provider prompt caching can hit; everything that changes per turn
    lives in build_state_block and travels as an input message instead."""
    objectives = "\n".join(
        f"- {s.field} ({s.type}): {s.objective}" for s in plan.steps)
    prohibited = ", ".join(plan.boundaries.prohibited_topics) or "none configured"
    one_at_a_time = "yes" if plan.boundaries.ask_one_question_at_a_time else "no"
    return f"""{plan.persona}

You are conducting an open but bounded screening interview. You choose every
substantive question and its wording; there is no scripted question order.

Evidence objectives:
{objectives}

Boundaries:
- Maximum caller turns: {plan.boundaries.max_turns}
- Maximum duration: {plan.boundaries.max_duration_minutes:g} minutes
- Ask one question at a time: {one_at_a_time}
- Prohibited topics: {prohibited}

Rules:
- Respond to the caller's latest meaning first. Evidence objectives are coverage
  goals, not a script, required order, or whitelist of allowed conversation.
- Handle questions, corrections, uncertainty, requests for clarification, and
  brief role-relevant conversation before naturally returning to screening.
- If the caller asks what you need from them, explain the relevant screening
  goal plainly instead of using a generic prompt or immediately re-asking it.
- When it is natural to continue collecting evidence, choose the most useful
  question yourself. It may clarify context rather than directly fill a field.
- Do not repeat established information. If part of a compound topic was
  answered, ask only for what remains missing.
- A hedged answer ("I guess so", "probably", "around eight") IS an answer.
  Accept it and move on; never ask the same objective again in the next turn.
- Ask ONE concise question at a time. Keep speech to one or two short sentences.
- Avoid empty acknowledgements such as "Okay", "Got it", or "Alright". Only
  acknowledge something when the acknowledgement adds useful meaning.
- ALWAYS provide a spoken reply for the caller. This request has no tools.
- Treat information in the latest caller message as already known when choosing
  your reply, even if it still appears in the missing-objectives state block.
  Never re-ask for information the caller just supplied.
- If the latest message appears to satisfy the final missing objective, close
  naturally without asking another question; the backend verifies coverage.
- If the caller asks to stop, speak only a brief goodbye. Do not ask another
  interview question.
- When every objective is covered, briefly close without another question.
- Never invent answers. Never promise pay, benefits, or hiring decisions.
- A system message states what is established and what remains; it never gives
  you a required question or question order."""


def build_extraction_instructions(plan) -> str:
    """Static prompt for the non-speaking provisional evidence pass."""
    objectives = "\n".join(
        f"- {s.field} ({s.type}): {s.objective}" for s in plan.steps)
    return f"""Extract provisional interview evidence. Produce no spoken reply.

Evidence objectives:
{objectives}

Rules:
- Call record_answer for every objective supported by the caller's verbatim
  words. Use the exact supported words as quote; never infer unsupported facts.
- The latest caller message may answer multiple objectives. Record each one.
- A hedged but substantive answer is evidence: record "I guess so" / "probably"
  as true, "around five" as 5, with the verbatim words as the quote. Booleans
  must be the strings true or false. Skip an objective only when the caller
  declined it or did not address it at all.
- Call end_call with candidate_requested only for an explicit request to end
  the call, interview, or screening.
- After recording evidence, call end_call with interview_complete only when all
  objectives are covered. Call it with limit_reached only when the supplied
  runtime budget says a limit is reached.
- Output tool calls only. Backend validation is authoritative."""


def build_state_block(state: InterviewState) -> str:
    plan = state.plan
    # Quotes are truncated in the PROMPT only (full text stays in state for
    # post-call verification) — the prompt must not grow with caller verbosity.
    filled = "\n".join(f"- {name}: {rec.value!r} (they said: \"{rec.quote[:60]}\")"
                       for name, rec in state.fields.items()) or "- none yet"
    remaining = "\n".join(
        f"- {s.field} ({s.type}): {s.objective}"
        for s in state.remaining) or "- none"
    turns_left = max(0, plan.boundaries.max_turns - state.caller_turn_count)
    seconds_left = max(
        0.0, plan.boundaries.max_duration_minutes * 60 - state.elapsed_seconds)
    if state.end_call_request is not None:
        objective = "Termination is already requested. Do not ask another question."
    elif turns_left == 0 or seconds_left == 0:
        objective = (
            "The interview limit is reached. Ask no question. Give a brief "
            "closing statement; the backend owns the lifecycle transition.")
    elif state.remaining:
        objective = (
            "Treat the missing evidence as coverage goals, not a questionnaire. "
            "Respond to the caller's latest intent first. When natural, continue "
            "with one useful question that you formulate yourself; it may answer "
            "or clarify the conversation before pursuing missing evidence.")
    else:
        objective = (
            "All evidence objectives are covered. Give a brief closing statement "
            "without another question; the backend owns lifecycle state.")
    return f"""Recorded so far:
{filled}
Evidence objectives still missing:
{remaining}

Runtime budget: {turns_left} caller turns and {seconds_left:.0f} seconds remain.

{objective}"""


def build_system_prompt(state: InterviewState) -> str:
    """Composition kept for tests and offline inspection; the live request
    sends the two halves separately for cacheability."""
    return build_instructions(state.plan) + "\n\n" + build_state_block(state)


def turn_input(state: InterviewState, user_text: str) -> list[dict[str, str]]:
    """Bounded history plus the latest caller message, without duplication.

    ReplyController journals committed caller turns immediately. Direct engine
    callers and speculative drafts do not, so they still need the explicit item.
    """
    history = state.recent_history(8)
    if (history and history[-1].get("role") == "user"
            and history[-1].get("content") == user_text):
        return history
    return history + [{"role": "user", "content": user_text}]


def cacheable_input(state: InterviewState, user_text: str) -> list[dict[str, str]]:
    """Static -> stable history -> volatile state -> newest turn.

    This is the cache-friendly layout (the per-turn state block used to sit
    FIRST, which caps any shared prefix at the instructions), and it puts the
    coverage state next to the turn it applies to. It does NOT buy prompt
    caching here, and the measurements say why: the stable prefix reaches only
    ~849 tokens after six turns, under the provider's ~1024-token threshold,
    and recent_history() is a SLIDING window - once it slides the prefix
    changes, so no prefix can ever stabilise. Reaching the threshold would mean
    padding the instructions or keeping unbounded history; neither is worth a
    cost optimisation that showed no latency benefit in a direct A/B.
    cached_tokens stays 0, measured over six live turns.
    """
    items = turn_input(state, user_text)
    newest = items[-1:] if items and items[-1].get("role") == "user" else []
    stable = items[:len(items) - len(newest)]
    return stable + [{"role": "system", "content": build_state_block(state)}] + newest
