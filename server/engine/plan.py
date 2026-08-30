"""Bounded interview objectives + per-call evidence state.

## How this works
"Plan is data": everything about WHAT the interview must establish — evidence
fields, objectives, boundaries, knockouts and weights — lives in plans/*.yaml.
The LLM owns question choice and wording; the engine owns scope and validation.

InterviewState is the source of truth for one conversation: recorded fields
(value + verbatim quote, coerced to the declared type), a validated end-call
request, and bounded history. There is deliberately no live question script:
coverage is the set of unanswered objectives, which the LLM may pursue in any
natural order. Live captures remain provisional:
only caller-utterance-verified post-call evidence may produce a score or knockout.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TYPES = {"bool", "str", "float", "list"}


_SCORING_RULES = {"min_full", "expected", "equals", "answered"}


@dataclass(frozen=True)
class Step:
    field: str
    type: str
    objective: str = ""
    knockout: dict | None = None
    scoring: dict | None = None  # rubric lives in the plan, not in code


@dataclass(frozen=True)
class InterviewBounds:
    max_turns: int = 15
    max_duration_minutes: float = 8.0
    ask_one_question_at_a_time: bool = True
    prohibited_topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class Analysis:
    """One configured post-call ADVISORY analysis (plan is data): the LLM
    writes it after the call; it is never an input to scores or knockouts."""
    id: str
    title: str
    instruction: str


@dataclass(frozen=True)
class InterviewPlan:
    persona: str
    consent: str
    steps: list[Step]
    weights: dict[str, float]
    scoring_version: str
    language: str = "en"
    analyses: tuple[Analysis, ...] = ()
    boundaries: InterviewBounds = InterviewBounds()


@dataclass(frozen=True)
class Recorded:
    value: Any
    quote: str


@dataclass(frozen=True)
class EndCallRequest:
    reason: str
    closing_message: str


END_CALL_REASONS = frozenset({
    "candidate_requested", "consent_refused", "interview_complete",
    "knockout", "max_duration", "max_turns", "limit_reached", "safety",
    "technical_failure",
})


def load_plan(path: Path) -> InterviewPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    steps = []
    for item in raw["steps"]:
        if "ask" in item:
            raise ValueError(
                f"step {item.get('field')!r}: scripted 'ask' is not allowed; "
                "describe the evidence objective instead")
        data = dict(item)
        data.setdefault("objective", str(data["field"]).replace("_", " "))
        steps.append(Step(**data))
    for s in steps:
        if s.type not in _TYPES:
            raise ValueError(f"step {s.field!r}: unknown type {s.type!r}")
        if s.scoring is not None and not set(s.scoring) <= _SCORING_RULES:
            raise ValueError(f"step {s.field!r}: unknown scoring rule "
                             f"{set(s.scoring) - _SCORING_RULES}")
    step_fields = {s.field for s in steps}
    weights = raw.get("weights", {})
    for w in weights:
        if w not in step_fields:
            raise ValueError(f"weights reference unknown field {w!r}")
    analyses = tuple(Analysis(id=str(a["id"]), title=str(a["title"]),
                              instruction=str(a["instruction"]))
                     for a in raw.get("analyses", []))
    ids = [a.id for a in analyses]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate analysis id in {sorted(ids)}")
    bounds_raw = raw.get("boundaries", {})
    boundaries = InterviewBounds(
        max_turns=int(bounds_raw.get("max_turns", 15)),
        max_duration_minutes=float(bounds_raw.get("max_duration_minutes", 8.0)),
        ask_one_question_at_a_time=bool(
            bounds_raw.get("ask_one_question_at_a_time", True)),
        prohibited_topics=tuple(str(v) for v in
                                  bounds_raw.get("prohibited_topics", [])),
    )
    if boundaries.max_turns < 1 or boundaries.max_duration_minutes <= 0:
        raise ValueError("interview boundaries must be positive")
    return InterviewPlan(
        persona=raw["persona"],
        consent=raw["consent"],
        steps=steps,
        weights=weights,
        scoring_version=raw["scoring_version"],
        language=raw.get("language", "en"),
        analyses=analyses,
        boundaries=boundaries,
    )


@lru_cache(maxsize=4)
def load_plan_cached(path_str: str) -> InterviewPlan:
    """Process-wide plan cache: plans are immutable data, loaded once. The app
    lifespan warms this at startup so a broken plan fails the boot, not call #1."""
    return load_plan(Path(path_str))


def _coerce(value: Any, type_name: str) -> Any:
    match type_name:
        case "bool":
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in ("true", "yes", "y", "1"):
                return True
            if normalized in ("false", "no", "n", "0"):
                return False
            if normalized in ("unknown", "unsure", "not sure", "ambiguous", ""):
                return None
            raise ValueError(f"ambiguous boolean {value!r}")
        case "float":
            if isinstance(value, (int, float)):
                return float(value)
            # Tolerate prose around the digits ("6.5 years", "about 3 or so"):
            # models narrate; the first numeric token is the answer.
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if match is None:
                raise ValueError(f"no number in {value!r}")
            return float(match.group())
        case "list":
            if isinstance(value, list):
                return [str(v).strip() for v in value]
            return [part.strip() for part in str(value).split(",") if part.strip()]
        case _:
            return str(value)


class InterviewState:
    def __init__(self, plan: InterviewPlan) -> None:
        self.plan = plan
        self.fields: dict[str, Recorded] = {}
        self.history: list[dict[str, str]] = []
        self.knocked_out: str | None = None
        self.end_call_request: EndCallRequest | None = None
        self.caller_turn_count = 0
        self.elapsed_seconds = 0.0
        # Audit trail of every tool call the model attempted: applied or not,
        # with the reason — orchestration and the post-call report read this.
        self.tool_ledger: list[dict] = []

    @property
    def done(self) -> bool:
        return self.next_needed is None

    @property
    def next_needed(self) -> Step | None:
        """First unanswered evidence objective, for compatibility/reporting."""
        return next((s for s in self.plan.steps if s.field not in self.fields), None)

    @property
    def remaining(self) -> list[Step]:
        return [s for s in self.plan.steps if s.field not in self.fields]

    def record(self, field: str, value: Any, quote: str) -> bool:
        step = next((s for s in self.plan.steps if s.field == field), None)
        if step is None:
            return False
        try:
            coerced = _coerce(value, step.type)
        except (TypeError, ValueError):
            return False
        if coerced is None:
            return False  # nullable/unknown stays unanswered; it never becomes false
        self.fields[field] = Recorded(value=coerced, quote=quote)
        # Live LLM tool captures are provisional. Only post-call caller-utterance
        # verification may produce a candidate knockout.
        return True

    def request_end_call(self, reason: str, closing_message: str) -> bool:
        """Validate a one-way termination request; first valid request wins."""
        reason = str(reason).strip()
        message = " ".join(str(closing_message).split())
        if (self.end_call_request is not None or reason not in END_CALL_REASONS
                or not message or len(message) > 240 or "?" in message):
            return False
        self.end_call_request = EndCallRequest(reason, message)
        return True

    def note_caller_turn(self) -> None:
        self.caller_turn_count += 1

    def update_elapsed(self, seconds: float) -> None:
        self.elapsed_seconds = max(0.0, float(seconds))

    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def recent_history(self, n: int) -> list[dict[str, str]]:
        return self.history[-n:]
