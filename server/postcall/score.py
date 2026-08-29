"""Deterministic scoring: verified evidence -> number. Phase 5a.

## How this works
The LLM never computes the score (CLAUDE.md hard rule) — extraction hands over
verified {value, quote, confidence} fields and everything below is arithmetic
on the plan's own rubric ("plan is data" extended to scoring):
- knockouts first: a disqualifying recorded value zeroes the call outright.
- each weighted field gets a subscore in [0,1] from its step's scoring rule:
  min_full (linear ramp to a threshold), expected (fraction of an expected
  list present), contains_any (keyword hit), answered (presence).
- score = sum(weight * subscore). Confidence below 0.5 on a weighted field, or
  a missing weighted field, sets needs_review with a stated reason — the number
  still computes, but a human is told not to trust it blindly.
Same inputs, same score, forever — scoring_version stamps which rubric did it.
"""
from dataclasses import dataclass, field as dc_field
from typing import Any

from server.engine.plan import InterviewPlan, Step
from server.postcall.extract import Extracted

CONFIDENCE_REVIEW_THRESHOLD = 0.5


@dataclass
class ScoreResult:
    score: float
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
        held = [str(v).lower() for v in (value if isinstance(value, list) else [value])]
        hits = sum(1 for exp in rule["expected"]
                   if any(str(exp).lower() in h for h in held))
        return hits / len(rule["expected"])
    if "contains_any" in rule:
        text = " ".join(value).lower() if isinstance(value, list) else str(value).lower()
        return 1.0 if any(str(k).lower() in text for k in rule["contains_any"]) else 0.0
    return 1.0 if value not in (None, "", []) else 0.0  # answered


def score_call(plan: InterviewPlan, extracted: dict[str, Extracted]) -> ScoreResult:
    result = ScoreResult(score=0.0, subscores={}, needs_review=False,
                         knocked_out=None, scoring_version=plan.scoring_version)
    for step in plan.steps:
        ext = extracted.get(step.field)
        if (step.knockout is not None and ext is not None
                and ext.value == step.knockout.get("equals")):
            result.knocked_out = step.field
            result.reasons.append(f"knockout: {step.field} = {ext.value!r}")
    total = 0.0
    for step in plan.steps:
        weight = plan.weights.get(step.field)
        if weight is None:
            continue
        ext = extracted.get(step.field)
        if ext is None or ext.value is None:
            sub, confidence = 0.0, 0.0
            result.needs_review = True
            result.reasons.append(f"{step.field}: not answered")
        else:
            sub, confidence = _subscore(step, ext.value), ext.confidence
            if confidence < CONFIDENCE_REVIEW_THRESHOLD:
                result.needs_review = True
                result.reasons.append(
                    f"{step.field}: low confidence ({confidence:.2f})")
        result.subscores[step.field] = {
            "subscore": sub, "weight": weight,
            "weighted": weight * sub, "confidence": confidence,
        }
        total += weight * sub
    result.score = 0.0 if result.knocked_out else round(total, 6)
    return result
