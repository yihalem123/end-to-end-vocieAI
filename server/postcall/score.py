"""Deterministic scoring: verified evidence -> number.

## How this works
The LLM never computes the score (a hard rule of this design) — extraction hands over
verified {value, quote, confidence} fields and everything below is arithmetic
on the plan's own rubric ("plan is data" extended to scoring):
- knockouts first: a disqualifying recorded value zeroes the call outright.
- each weighted field gets a subscore in [0,1] from its step's scoring rule:
  min_full (linear ramp), expected (exact normalized list membership), equals,
  or answered.
- unverified, low-confidence, unknown, contradictory, or missing material
  evidence cannot trigger a knockout or contribute to a numeric score. Such a
  call has score=None and requires human review.
Same inputs, same score, forever — scoring_version stamps which rubric did it.
"""
import re
from dataclasses import dataclass, field as dc_field
from typing import Any

from server.engine.plan import InterviewPlan, Step
from server.postcall.extract import Extracted

CONFIDENCE_REVIEW_THRESHOLD = 0.5


@dataclass
class ScoreResult:
    score: float | None
    subscores: dict[str, dict]
    needs_review: bool
    knocked_out: str | None
    scoring_version: str
    reasons: list[str] = dc_field(default_factory=list)


def _subscore(step: Step, value: Any) -> float:
    rule = step.scoring or {"answered": True}
    if "min_full" in rule:
        return min(1.0, max(0.0, float(value) / float(rule["min_full"])))
    if "expected" in rule:
        held = {_normalized(v) for v in (value if isinstance(value, list) else [value])}
        hits = sum(1 for exp in rule["expected"] if _normalized(exp) in held)
        return hits / len(rule["expected"])
    if "equals" in rule:
        return 1.0 if value == rule["equals"] else 0.0
    return 1.0 if value not in (None, "", []) else 0.0  # answered


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _ineligible_reason(field: str, ext: Extracted | None) -> str | None:
    if ext is None or ext.value is None:
        return f"{field}: not answered"
    if not ext.verified:
        return f"{field}: evidence not verified to a caller utterance"
    if ext.contradictory:
        return f"{field}: contradictory evidence"
    if ext.confidence < CONFIDENCE_REVIEW_THRESHOLD:
        return f"{field}: low confidence ({ext.confidence:.2f})"
    return None


def score_call(plan: InterviewPlan, extracted: dict[str, Extracted]) -> ScoreResult:
    result = ScoreResult(score=0.0, subscores={}, needs_review=False,
                         knocked_out=None, scoring_version=plan.scoring_version)
    for step in plan.steps:
        ext = extracted.get(step.field)
        reason = _ineligible_reason(step.field, ext)
        if step.knockout is not None and reason is not None:
            result.needs_review = True
            result.reasons.append(reason)
        if (step.knockout is not None and reason is None
                and ext is not None and ext.value == step.knockout.get("equals")):
            result.knocked_out = step.field
            result.needs_review = True
            result.reasons.append(f"knockout: {step.field} = {ext.value!r}")
    total = 0.0
    for step in plan.steps:
        weight = plan.weights.get(step.field)
        if weight is None:
            continue
        ext = extracted.get(step.field)
        reason = _ineligible_reason(step.field, ext)
        if reason is not None:
            sub, confidence = 0.0, 0.0
            result.needs_review = True
            if reason not in result.reasons:
                result.reasons.append(reason)
        else:
            assert ext is not None
            sub, confidence = _subscore(step, ext.value), ext.confidence
        result.subscores[step.field] = {
            "subscore": sub, "weight": weight,
            "weighted": weight * sub, "confidence": confidence,
        }
        total += weight * sub
    result.score = (0.0 if result.knocked_out else
                    None if result.needs_review else round(total, 6))
    return result
