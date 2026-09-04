"""Call-report HTML view: a view-model over the stored report, rendered by a
Jinja2 template with autoescaping (templates/report.html).

## How this works
Python decides WHAT to show; the template decides HOW. build_view() reduces
the report dict to plain values - verdict tile (four mutually exclusive
states: scored / knocked out / needs review / not scored), stat tiles, the
call-replay timeline with every extraction chip anchored to the caller
utterance that produced it, score-breakdown arithmetic, advisory AI notes,
the tool-call audit trail, and latency by stage. Colours are CSS-variable
names so the page re-themes from one checkbox (body:has(#themeToggle:checked))
with no JavaScript at all - a test pins that the output contains no <script>.
Every dynamic string is escaped by the template engine, not by hand: a caller
utterance containing markup renders as text (tested).
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES)),
    autoescape=select_autoescape(default=True, default_for_string=True),
    trim_blocks=True, lstrip_blocks=True,
)

_TONES = {
    "good": ("var(--tile-good-bg)", "var(--acc-border)", "var(--acc-strong)", "var(--good)"),
    "bad": ("var(--tile-bad-bg)", "var(--bad-border)", "var(--bad)", "var(--bad)"),
    "warn": ("var(--tile-warn-bg)", "var(--warn-border)", "var(--warn)", "var(--warn)"),
    "mute": ("var(--sub-card)", "var(--line)", "var(--dim2)", "var(--faint)"),
}
_CHIP_COLORS = {
    "ok": ("var(--acc-bg)", "var(--acc-border)", "var(--acc)", "var(--acc-strong)", "var(--good)"),
    "warn": ("var(--warn-bg)", "var(--warn-border)", "var(--warn)", "var(--dim)", "var(--warn)"),
    "bad": ("var(--bad-bg)", "var(--bad-border)", "var(--bad)", "var(--acc-strong)", "var(--bad)"),
}
_STAGE_LABELS = (
    # Ordered as the caller experiences them; turn latency is the whole wait.
    ("endpoint_delay_ms", "endpoint delay", "vad stop → commit"),
    ("llm_ttft_ms", "llm ttft", "commit → first token"),
    ("tts_ttfb_ms", "tts ttfb", "sentence → first audio byte"),
    ("first_audio_ms", "first audio", "commit → first frame sent"),
    ("turn_latency_ms", "turn latency", "vad stop → first frame sent"),
)


def _verdict(report: dict) -> dict:
    knocked = report.get("knocked_out")
    score = report.get("score")
    if knocked:
        label, big, sub, tone = "Knocked out", "—", str(knocked), "bad"
    elif score is not None:
        sub = ("needs review" if report.get("needs_review")
               else str(report.get("scoring_version") or ""))
        label, big, tone = "Score", f"{score:.2f}", "good"
    elif report.get("needs_review"):
        label, big, sub, tone = "Needs review", "· ·", "score withheld", "warn"
    else:
        reasons = report.get("reasons") or [str(report.get("session_state", ""))]
        label, big, sub, tone = "Not scored", "—", str(reasons[0]), "mute"
    tile_bg, tile_border, tile_label, sub_col = _TONES[tone]
    return {"label": label, "big": big, "sub": sub, "tile_bg": tile_bg,
            "tile_border": tile_border, "tile_label": tile_label, "sub_col": sub_col}


def _chip(name: str, f: dict, knocked: str | None) -> dict:
    conf = float(f.get("confidence") or 0.0)
    if knocked == name:
        kind, label, status = "bad", "Knockout", "✗ disqualifying"
    elif f.get("contradictory"):
        kind, label, status = "warn", "Flagged", f"⚠ contradictory {conf:.2f}"
    elif not f.get("verified"):
        kind, label, status = "warn", "Flagged", f"⚠ unverified {conf:.2f}"
    else:
        kind, label, status = "ok", "Extracted", f"✓ verified {conf:.2f}"
    bg, border, label_col, val_col, status_col = _CHIP_COLORS[kind]
    return {"bg": bg, "border": border, "label_col": label_col, "label": label,
            "val_col": val_col, "name": name, "value": str(f.get("value")),
            "status_col": status_col, "status": status}


def _timeline(report: dict) -> list[dict]:
    knocked = report.get("knocked_out")
    by_utterance: dict[str, list] = {}
    for name, f in report.get("fields", {}).items():
        if f.get("utterance_id"):
            by_utterance.setdefault(str(f["utterance_id"]), []).append((name, f))
    rows = []
    for entry in report.get("conversation", []):
        agent = entry.get("role") == "agent"
        chips = [_chip(n, f, knocked) for n, f in
                 by_utterance.get(str(entry.get("utterance_id", "")), [])]
        rows.append({
            "who": "Asandia" if agent else "Candidate",
            "who_col": "var(--acc-text)" if agent else "var(--dim2)",
            "dot": "var(--acc)" if chips else "var(--dot-idle)",
            "glow": "var(--acc-glow)" if chips else "none",
            "text_col": "var(--strong)" if chips else "var(--dim)",
            "text": str(entry.get("text", "")),
            "interrupted": bool(entry.get("interrupted")),
            "chips": chips,
        })
    return rows


def _unanchored(report: dict) -> list[dict]:
    """Evidence that could not be anchored to a conversation utterance must
    still be visible (quarantined goldens, missing ids): lossless fallback."""
    anchored = {str(e.get("utterance_id")) for e in report.get("conversation", [])
                if e.get("utterance_id")}
    knocked = report.get("knocked_out")
    return [_chip(n, f, knocked) for n, f in report.get("fields", {}).items()
            if str(f.get("utterance_id")) not in anchored]


def _breakdown(report: dict) -> list[dict]:
    subs = report.get("subscores", {})
    if not subs or report.get("score") is None:
        return []
    max_weight = max((s.get("weight", 0) for s in subs.values()), default=1) or 1
    return [{
        "name": name,
        "pct": round(s.get("weighted", 0) / max_weight * 100),
        "math": (f'{s.get("subscore", 0):.2f} × {s.get("weight", 0):.2f} '
                 f'= {s.get("weighted", 0):.3f}'),
    } for name, s in subs.items()]


def _audit(report: dict) -> list[dict]:
    rows = []
    for a in report.get("tool_ledger", []):
        applied = a.get("applied")
        args = a.get("arguments") or {}
        rows.append({
            "col": "var(--good)" if applied else "var(--bad)",
            "status": "Applied" if applied else "Rejected",
            "tool": f'{a.get("name")}({args.get("field", "")})',
            "reason": str(a.get("reason")
                          or f'turn {a.get("turn_id")} · generation {a.get("generation_id")}'),
        })
    return rows


def _latency(report: dict) -> list[dict]:
    """Per-stage p50/p95, so turn latency can be decomposed, not just quoted."""
    stages = report.get("latency") or {}
    rows = []
    for key, label, meaning in _STAGE_LABELS:
        for prefix in ("", "flux_"):
            stat = stages.get(prefix + key)
            if not stat:
                continue
            rows.append({
                "name": label + " (flux)" if prefix else label,
                "meaning": meaning,
                "weight": "600" if key == "turn_latency_ms" else "400",
                "p50": f'{stat["p50"]:.0f}', "p95": f'{stat["p95"]:.0f}',
            })
    return rows


def build_view(report: dict) -> dict:
    fields = report.get("fields", {})
    delays = report.get("endpoint_delays_ms", [])
    state = str(report.get("session_state", ""))
    state_ok = state == "completed"
    ko_count = 1 if report.get("knocked_out") else 0
    return {
        "call_id": str(report.get("call_id", "")),
        "created_at": str(report.get("created_at", "")),
        "turn_count": report.get("turn_count", 0),
        "verdict": _verdict(report),
        "state": {
            "name": state,
            "col": ("var(--good)" if state_ok
                    else ("var(--bad)" if state == "failed" else "var(--warn)")),
            "bg": "var(--good-bg)" if state_ok else "var(--warn-bg)",
            "border": "var(--good-border)" if state_ok else "var(--warn-border)",
        },
        "stats": {
            "verified": sum(1 for f in fields.values() if f.get("verified")),
            "total": len(fields),
            "knockouts": ko_count,
            "ko_col": "var(--bad)" if ko_count else "var(--good)",
            "median_delay": sorted(delays)[len(delays) // 2] if delays else 0,
        },
        "flags": "; ".join(str(r) for r in (report.get("reasons") or [])),
        "timeline": _timeline(report),
        "unanchored": _unanchored(report),
        "breakdown": _breakdown(report),
        "ai_notes": [{"title": str(a.get("title", "")), "text": str(a.get("text", ""))}
                     for a in (report.get("analyses") or [])],
        "audit": _audit(report),
        "latency": _latency(report),
        "delays": [{"ms": ms,
                    "col": "var(--warn)" if ms > 700 else "var(--dim)",
                    "border": "var(--warn-border)" if ms > 700 else "var(--line-soft)"}
                   for ms in delays],
    }


def render_report_html(report: dict) -> str:
    template = _env.get_template("report.html")
    return template.render(chip_html=_env.get_template("_chip.html").module.chip_html,
                           **build_view(report))
