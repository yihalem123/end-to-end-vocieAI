"""Post-call evidence extraction: transcript -> {value, quote, confidence}. Phase 5a.

## How this works
After the call ends we re-extract every plan field from the FULL transcript in
one Responses API call (gpt-5.6-terra, Structured Outputs strict:true — the
schema is built from the plan, so "plan is data" holds here too). The live
engine's record_answer captures are a real-time convenience; this pass is the
authoritative one because it sees the whole conversation and every correction.

Then the part the LLM is not trusted with: verify_and_coerce() checks each
QUOTE against the transcript with normalized matching (case, punctuation and
whitespace insensitive). A quote that does not appear in the transcript zeroes
that field's confidence — the model cannot manufacture evidence (CLAUDE.md hard
rule). Values are coerced to the step's declared type; a value that will not
coerce also zeroes confidence. score.py consumes only these verified fields.
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


def render_transcript(entries: list[dict]) -> str:
    """entries: {"role": "caller"|"agent", "text": ..., "interrupted": bool?}"""
    lines = []
    for e in entries:
        speaker = "CALLER" if e["role"] == "caller" else "AGENT"
        suffix = " [cut off by caller]" if e.get("interrupted") else ""
        lines.append(f"{speaker}: {e['text']}{suffix}")
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
    """Strict Structured Outputs schema: one {value, quote, confidence} per field."""
    def per_field(step) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "quote", "confidence"],
            "properties": {
                "value": {"type": ["string", "null"],
                          "description": _VALUE_DESCRIPTIONS[step.type]},
                "quote": {"type": ["string", "null"],
                          "description": "VERBATIM caller words supporting the value."},
                "confidence": {"type": "number",
                               "description": "0-1: how clearly the transcript supports it."},
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


def verify_and_coerce(raw: dict, transcript: str, plan: InterviewPlan) -> dict[str, Extracted]:
    """The deterministic trust boundary: quotes must exist in the transcript."""
    haystack = _normalize(transcript)
    out: dict[str, Extracted] = {}
    for step in plan.steps:
        entry = raw.get(step.field) or {}
        value, quote = entry.get("value"), entry.get("quote") or ""
        confidence = max(0.0, min(1.0, float(entry.get("confidence") or 0.0)))
        if value is None:
            continue  # never answered: absent, not zero-confidence noise
        if not quote or _normalize(quote) not in haystack:
            confidence = 0.0  # unverifiable evidence => worthless evidence
        try:
            coerced = _coerce(value, step.type)
        except (TypeError, ValueError):
            coerced, confidence = None, 0.0
        out[step.field] = Extracted(value=coerced, quote=quote, confidence=confidence)
    return out


async def extract_call(settings: Settings, plan: InterviewPlan,
                       transcript: str) -> dict[str, Extracted]:
    body = {
        "model": settings.extract_model,
        "instructions": (
            "Extract the screening answers from this call transcript. For every "
            "field: the answer, the caller's verbatim supporting words, and your "
            "confidence. Use null for anything the caller never answered. Never "
            "paraphrase quotes — copy the caller's words exactly."
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
    return verify_and_coerce(json.loads(text), transcript, plan)
