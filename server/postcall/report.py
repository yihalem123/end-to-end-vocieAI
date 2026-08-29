"""Post-call pipeline trigger + in-memory report store. Phase 5b.

## How this works
When a call with any conversation ends, app.py fires run_postcall() as a
background task: render the transcript -> extract_call (terra, strict schema)
-> verify quotes -> score deterministically -> store the whole thing under the
call id. The store is a bounded in-process dict (newest-first eviction) — a
database is a production talking point, not a demo requirement. build_report()
is pure so the report shape is unit-testable without the vendor call; the HTML
view is a deliberately plain table (the report is the product, not the CSS).
"""
import logging
import time
from collections import OrderedDict
from html import escape
from pathlib import Path

from server.config import Settings
from server.engine.plan import load_plan
from server.postcall.extract import Extracted, extract_call, render_transcript
from server.postcall.score import ScoreResult, score_call

log = logging.getLogger(__name__)

MAX_REPORTS = 50
reports: OrderedDict[str, dict] = OrderedDict()


def build_report(call_id: str, conversation: list[dict], turns: list,
                 extracted: dict[str, Extracted], result: ScoreResult) -> dict:
    return {
        "call_id": call_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": result.score,
        "needs_review": result.needs_review,
        "knocked_out": result.knocked_out,
        "scoring_version": result.scoring_version,
        "reasons": result.reasons,
        "subscores": result.subscores,
        "fields": {
            name: {"value": e.value, "quote": e.quote, "confidence": e.confidence}
            for name, e in extracted.items()
        },
        "turn_count": len(turns),
        "endpoint_delays_ms": [round(t.endpoint_delay * 1000) for t in turns],
        "conversation": conversation,
    }


async def run_postcall(call_id: str, conversation: list[dict], turns: list,
                       settings: Settings) -> None:
    try:
        plan = load_plan(Path(settings.plan_path))
        transcript = render_transcript(conversation)
        extracted = await extract_call(settings, plan, transcript)
        result = score_call(plan, extracted)
        reports[call_id] = build_report(call_id, conversation, turns,
                                        extracted, result)
        while len(reports) > MAX_REPORTS:
            reports.popitem(last=False)
        log.info("postcall %s: score=%.2f needs_review=%s knocked_out=%s",
                 call_id, result.score, result.needs_review, result.knocked_out)
    except Exception:
        log.exception("postcall pipeline failed for %s", call_id)
        reports[call_id] = {"call_id": call_id, "error": "postcall failed — see logs",
                            "conversation": conversation}


def render_html(report: dict) -> str:
    if "error" in report:
        return f"<h1>Call {escape(report['call_id'])}</h1><p>{escape(report['error'])}</p>"
    fields = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(str(f['value']))}</td>"
        f"<td>{f['confidence']:.2f}</td><td>{escape(f['quote'] or '')}</td></tr>"
        for name, f in report["fields"].items())
    subs = "".join(
        f"<tr><td>{escape(name)}</td><td>{s['subscore']:.2f}</td>"
        f"<td>{s['weight']:.2f}</td><td>{s['weighted']:.2f}</td></tr>"
        for name, s in report["subscores"].items())
    convo = "".join(
        f"<p><b>{escape(e['role'])}:</b> {escape(e['text'])}"
        f"{' ⏹' if e.get('interrupted') else ''}</p>"
        for e in report["conversation"])
    verdict = ("KNOCKED OUT: " + report["knocked_out"] if report["knocked_out"]
               else f"score {report['score']:.2f}"
                    + (" — needs review" if report["needs_review"] else ""))
    return f"""<!doctype html><meta charset="utf-8">
<title>Call report {escape(report['call_id'])}</title>
<style>body{{font-family:system-ui;max-width:44rem;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #ddd;padding:.3rem .6rem;text-align:left}}</style>
<h1>Call {escape(report['call_id'])}</h1>
<p><b>{escape(verdict)}</b> · {escape(report['scoring_version'])} · {escape(report['created_at'])}</p>
<p>{escape('; '.join(report['reasons']) or 'no flags')}</p>
<h2>Fields</h2><table><tr><th>field</th><th>value</th><th>conf</th><th>quote</th></tr>{fields}</table>
<h2>Score breakdown</h2><table><tr><th>field</th><th>subscore</th><th>weight</th><th>weighted</th></tr>{subs}</table>
<p>turns: {report['turn_count']} · endpoint delays (ms): {report['endpoint_delays_ms']}</p>
<h2>Transcript</h2>{convo}"""
