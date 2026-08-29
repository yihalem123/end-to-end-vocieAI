"""Plan loading + InterviewState. Phase 4.

## How this works
"Plan is data" (CLAUDE.md): everything about WHAT the interview covers — steps,
field types, knockouts, weights — lives in plans/*.yaml and is validated at
load. The engine walks it; the LLM only phrases.

InterviewState is the source of truth for one conversation: recorded fields
(value + verbatim quote, coerced to the step's declared type), the step cursor,
and bounded history. Advancement is LLM-signaled but ENGINE-VALIDATED:
request_advance() refuses unless the current step's field is recorded, then
moves the cursor forward past any step already filled by volunteered info. The
LLM can propose flow; it cannot skip coverage. Live captures remain provisional:
only caller-utterance-verified post-call evidence may produce a score or knockout.
"""
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml

_TYPES = {"bool", "str", "float", "list"}


_SCORING_RULES = {"min_full", "expected", "equals", "answered"}


@dataclass(frozen=True)
class Step:
    field: str
    type: str
    ask: str | None = None
    knockout: dict | None = None
    scoring: dict | None = None  # rubric lives in the plan, not in code


@dataclass(frozen=True)
class InterviewPlan:
    persona: str
    consent: str
    steps: list[Step]
    weights: dict[str, float]
    scoring_version: str
    language: str = "en"


@dataclass(frozen=True)
class Recorded:
    value: Any
    quote: str


def load_plan(path: Path) -> InterviewPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    steps = [Step(**s) for s in raw["steps"]]
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
    return InterviewPlan(
        persona=raw["persona"],
        consent=raw["consent"],
        steps=steps,
        weights=weights,
        scoring_version=raw["scoring_version"],
        language=raw.get("language", "en"),
    )


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
        self.step_idx = 0
        self.knocked_out: str | None = None

    @property
    def current_step(self) -> Step:
        return self.plan.steps[min(self.step_idx, len(self.plan.steps) - 1)]

    @property
    def done(self) -> bool:
        return self.step_idx >= len(self.plan.steps)

    @property
    def next_needed(self) -> Step | None:
        """First step with no recorded answer — tracks NEED, not the cursor.
        The prompt targets need so a model that records answers without ever
        signaling advance_step still gets pointed at the right question."""
        return next((s for s in self.plan.steps if s.field not in self.fields), None)

    @property
    def next_askable(self) -> Step | None:
        """First unfilled step that has words to ask with. Ask-less steps
        (knockout companions like rn_license_active) are filled as side effects
        of other questions and can't be the spoken objective themselves."""
        for i, s in enumerate(self.plan.steps):
            if s.field in self.fields:
                continue
            if s.ask is not None or i == 0:  # step 0 asks via plan.consent
                return s
        return None

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

    def request_advance(self) -> bool:
        """LLM proposes advancement; the engine grants it only when the current
        step is actually covered, then skips past anything already filled."""
        if self.done or self.current_step.field not in self.fields:
            return False
        self.step_idx += 1
        while not self.done and self.current_step.field in self.fields:
            self.step_idx += 1
        return True

    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def recent_history(self, n: int) -> list[dict[str, str]]:
        return self.history[-n:]
