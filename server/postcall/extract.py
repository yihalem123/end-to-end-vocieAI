"""Post-call evidence extraction: caller utterances -> verified typed fields.

## How this works
After the call ends we re-extract every plan field from the FULL transcript in
one Responses API call (gpt-5.6-terra, Structured Outputs strict:true — the
schema is built from the plan, so "plan is data" holds here too). The live
engine's record_answer captures are a real-time convenience; this pass is the
authoritative one because it sees the whole conversation and every correction.

Then the deterministic trust boundary checks each quote against the one caller
utterance id named by the extraction. Agent words are never evidence. Missing,
fabricated, ambiguous, low-confidence, or contradictory evidence remains visible
in the report but score.py cannot use it for a score or knockout.
"""
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import Settings
from server.engine.plan import InterviewPlan, _coerce

RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class Extracted:
    value: Any
    quote: str
    confidence: float
    utterance_id: str | None = None
    verified: bool = False
    contradictory: bool = False


def render_transcript(entries: list[dict]) -> str:
    """entries: {"role": "caller"|"agent", "text": ..., "interrupted": bool?}"""
    lines = []
    for e in entries:
        speaker = "CALLER" if e["role"] == "caller" else "AGENT"
        identity = (f" [{e['utterance_id']}]" if e["role"] == "caller"
                    and e.get("utterance_id") else "")
        suffix = " [cut off by caller]" if e.get("interrupted") else ""
        lines.append(f"{speaker}{identity}: {e['text']}{suffix}")
    return "\n".join(lines)


_VALUE_DESCRIPTIONS = {
    # Type-aware so the model writes machine-readable values (golden-run
    # regression: a float field came back as "six and a half").
    "float": "The numeric answer in digits (e.g. 6.5); null if never answered.",
    "bool": "true or false (as a string); null if never answered.",
    "list": "Comma-separated items; null if never answered.",
    "str": "The answer as short text; null if never answered.",
}


def build_schema(plan: InterviewPlan) -> dict:
    """Strict schema with value, caller quote/id, confidence, and contradiction."""
    def per_field(step) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "quote", "utterance_id", "confidence",
                         "contradictory"],
            "properties": {
                "value": {"type": ["string", "null"],
                          "description": _VALUE_DESCRIPTIONS[step.type]},
                "quote": {"type": ["string", "null"],
                           "description": "VERBATIM caller words supporting the value."},
                "utterance_id": {"type": ["string", "null"],
                                 "description": "The CALLER utterance id containing the quote."},
                "confidence": {"type": "number",
                                "description": "0-1: how clearly the transcript supports it."},
                "contradictory": {"type": "boolean",
                                  "description": "True if caller evidence conflicts or is corrected ambiguously."},
            },
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [s.field for s in plan.steps],
        "properties": {s.field: per_field(s) for s in plan.steps},
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def verify_and_coerce(raw: dict, entries: list[dict],
                      plan: InterviewPlan) -> dict[str, Extracted]:
    """Trust boundary: evidence must name a caller utterance containing the quote."""
    caller_utterances = {
        str(entry["utterance_id"]): _normalize(str(entry.get("text", "")))
        for entry in entries
        if entry.get("role") == "caller" and entry.get("utterance_id")
    }
    out: dict[str, Extracted] = {}
    for step in plan.steps:
        entry = raw.get(step.field) or {}
        value, quote = entry.get("value"), entry.get("quote") or ""
        utterance_id = entry.get("utterance_id")
        utterance_key = str(utterance_id) if utterance_id is not None else None
        contradictory = bool(entry.get("contradictory", False))
        confidence = max(0.0, min(1.0, float(entry.get("confidence") or 0.0)))
        if value is None:
            continue  # never answered: absent, not zero-confidence noise
        verified = bool(
            quote and utterance_key in caller_utterances
            and _normalize(quote) in caller_utterances[utterance_key]
        )
        if not verified:
            confidence = 0.0
        try:
            coerced = _coerce(value, step.type)
        except (TypeError, ValueError):
            coerced, confidence = None, 0.0
        out[step.field] = Extracted(
            value=coerced, quote=quote, confidence=confidence,
            utterance_id=utterance_key,
            verified=verified, contradictory=contradictory,
        )
    return out


async def extract_call(settings: Settings, plan: InterviewPlan,
                       entries: list[dict]) -> dict[str, Extracted]:
    transcript = render_transcript(entries)
    body = {
        "model": settings.extract_model,
        "instructions": (
            "Extract the screening answers from this call transcript. For every "
            "field: the answer, the caller's verbatim supporting words, and your "
            "confidence, the caller utterance id containing that quote, and whether "
            "the caller gave contradictory evidence. Use null for unanswered fields. "
            "Never use AGENT words as evidence or paraphrase caller quotes."
        ),
        "input": transcript,
        "text": {"format": {"type": "json_schema", "name": "screening_extraction",
                            "strict": True, "schema": build_schema(plan)}},
        "store": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            RESPONSES_URL, json=body,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
    text = next(
        content["text"]
        for item in data["output"] if item["type"] == "message"
        for content in item["content"] if content["type"] == "output_text"
    )
    return verify_and_coerce(json.loads(text), entries, plan)
