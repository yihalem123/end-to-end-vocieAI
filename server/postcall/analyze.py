"""Configured post-call ADVISORY analyses. Additive to Phase 5's pipeline.

## How this works
The plan's `analyses:` block defines what gets written after each call (plan is
data — a dental-office plan can configure entirely different notes). One
Responses API call (the extract model) produces every configured analysis at
once via a strict schema built from the plan: one string per analysis id, each
described by its configured instruction. These are ADVISORY by construction:
they are rendered in their own clearly-labeled report section, and score.py
never sees them — the witness/judge boundary stays intact. A failure here
degrades to an absent section, never a failed report.
"""
import json
import logging

import httpx

from server.config import Settings
from server.engine.plan import InterviewPlan
from server.postcall.extract import RESPONSES_URL, render_transcript

log = logging.getLogger(__name__)


def build_analysis_schema(plan: InterviewPlan) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [a.id for a in plan.analyses],
        "properties": {
            a.id: {"type": "string", "description": a.instruction}
            for a in plan.analyses
        },
    }


def shape_analyses(plan: InterviewPlan, raw: dict) -> list[dict]:
    """Pair model output with configured titles, preserving plan order."""
    out = []
    for a in plan.analyses:
        text = str(raw.get(a.id) or "").strip()
        if text:
            out.append({"id": a.id, "title": a.title, "text": text})
    return out


async def analyze_call(settings: Settings, plan: InterviewPlan,
                       entries: list[dict]) -> list[dict]:
    if not plan.analyses:
        return []
    body = {
        "model": settings.extract_model,
        "instructions": (
            "You are writing ADVISORY post-call notes for a recruiter reviewing "
            "a screening call transcript. Produce each requested note per its "
            "description. Ground every observation in what was actually said. "
            "Do not assign scores, ratings, or hiring decisions."
        ),
        "input": render_transcript(entries),
        "text": {"format": {"type": "json_schema", "name": "postcall_analyses",
                            "strict": True,
                            "schema": build_analysis_schema(plan)}},
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
    return shape_analyses(plan, json.loads(text))
