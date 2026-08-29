"""Call-report HTML view, per the "Call Report" design canvas.

## How this works
Pure presentation over the stored report dict: a verdict tile with four
mutually exclusive states (scored / knocked out / needs review / not scored),
identity + stat tiles, then the centerpiece — a call-replay timeline where
every extraction chip is anchored to the caller utterance that produced it
(via the utterance_id the evidence pipeline already records). Score-breakdown
bars show the arithmetic (subscore × weight); the audit trail (tool ledger)
and per-turn endpoint delays collapse behind native <details> — no JS at all.
Every dynamic string is escaped.
"""
from html import escape

_TONES = {
    "good": ("linear-gradient(160deg,rgba(124,140,248,.16),rgba(124,140,248,.04))",
             "rgba(124,140,248,.35)", "#c3caff", "#6ee787"),
    "bad": ("linear-gradient(160deg,rgba(229,72,77,.14),rgba(229,72,77,.03))",
            "rgba(229,72,77,.4)", "#f28488", "#f28488"),
    "warn": ("linear-gradient(160deg,rgba(229,168,72,.12),rgba(229,168,72,.03))",
             "rgba(229,168,72,.4)", "#e5b96d", "#e5b96d"),
    "mute": ("rgba(255,255,255,.03)", "rgba(255,255,255,.1)",
             "rgba(232,233,240,.5)", "rgba(232,233,240,.45)"),
}

_CHIP_COLORS = {
    "ok": ("rgba(124,140,248,.08)", "rgba(124,140,248,.3)", "#7c8cf8",
           "#c3caff", "#6ee787"),
    "warn": ("rgba(229,168,72,.06)", "rgba(229,168,72,.3)", "#e5b96d",
             "rgba(232,233,240,.7)", "#e5b96d"),
    "bad": ("rgba(229,72,77,.07)", "rgba(229,72,77,.35)", "#f28488",
            "#c3caff", "#f28488"),
}

_CSS = """
* { margin:0; padding:0; box-sizing:border-box }
body { background:#0a0b10; color:#e8e9f0; font-family:'Space Grotesk',system-ui,sans-serif }
.wrap { display:flex; justify-content:center; padding:28px 32px 80px }
.col { width:1080px; max-width:100%; display:flex; flex-direction:column; gap:16px }
.grid { display:grid; grid-template-columns:200px 1fr 1fr 1fr; gap:14px }
.card { background:#0d0e15; border:1px solid rgba(255,255,255,.07); border-radius:14px }
.label { font:600 12px 'Space Grotesk'; letter-spacing:.08em; text-transform:uppercase;
         color:rgba(232,233,240,.5) }
.score-tile { grid-row:span 2; border-radius:14px; padding:20px; display:flex;
              flex-direction:column; justify-content:space-between; border:1px solid;
              min-height:170px }
.score-big { font:700 52px/1 'Space Grotesk'; color:#f2f3fa; letter-spacing:-.03em }
.id-card { grid-column:span 3; padding:16px 20px; display:flex; align-items:center;
           justify-content:space-between; gap:16px }
.mark { width:26px; height:26px; border-radius:8px;
        background:linear-gradient(135deg,#7c8cf8,#4c5ae0); flex:none }
.stat { padding:14px 18px; display:flex; flex-direction:column; gap:3px }
.stat b { font:600 26px 'Space Grotesk' }
.stat small { font-size:15px; color:rgba(232,233,240,.4) }
.state-chip { padding:6px 14px; border-radius:100px; font:500 12.5px 'Space Grotesk';
              border:1px solid; flex:none }
.tl-wrap { position:relative; padding-left:34px; display:flex; flex-direction:column;
           gap:22px }
.tl-line { position:absolute; left:11px; top:6px; bottom:6px; width:2px;
           background:linear-gradient(180deg,rgba(124,140,248,.5),rgba(124,140,248,.08)) }
.tl-row { position:relative }
.tl-dot { position:absolute; left:-30px; top:4px; width:10px; height:10px;
          border-radius:50%; border:2px solid #0a0b10 }
.tl-who { font:600 10.5px 'Space Grotesk'; letter-spacing:.08em;
          text-transform:uppercase; margin-bottom:4px }
.intr { color:#e5b96d; text-transform:none; letter-spacing:0 }
.tl-text { font:400 14.5px/1.55 'Space Grotesk'; max-width:720px }
.chip { margin-top:10px; max-width:720px; padding:12px 16px; border-radius:12px;
        border:1px solid; display:flex; align-items:center; gap:16px; flex-wrap:wrap }
.chip-label { font:600 11px 'Space Grotesk'; letter-spacing:.08em; text-transform:uppercase }
.chip-val { font:500 12.5px ui-monospace,Menlo,monospace }
.chip-status { font:500 12px 'Space Grotesk' }
.pad { padding:22px 32px; display:flex; flex-direction:column; gap:14px }
.card-head { display:flex; align-items:baseline; justify-content:space-between }
.sigma { font:600 13px ui-monospace,Menlo,monospace; color:#a3aefb }
.bk-row { display:grid; grid-template-columns:180px 1fr 170px; align-items:center; gap:14px }
.bk-name { font:500 12.5px ui-monospace,Menlo,monospace; color:rgba(232,233,240,.6) }
.bk-track { height:20px; border-radius:6px; background:rgba(255,255,255,.05);
            overflow:hidden; position:relative }
.bk-fill { position:absolute; inset:0;
           background:linear-gradient(90deg,rgba(124,140,248,.85),rgba(124,140,248,.45));
           border-radius:6px }
.bk-math { font:500 12px ui-monospace,Menlo,monospace; color:rgba(232,233,240,.55);
           text-align:right }
details.card summary { list-style:none; cursor:pointer; padding:18px 32px;
                       display:flex; justify-content:space-between }
details.card summary::-webkit-details-marker { display:none }
details.card summary::after { content:"▾"; color:rgba(232,233,240,.4); font-size:14px }
details[open].card summary::after { content:"▴" }
.au-body { padding:0 32px 20px; display:flex; flex-direction:column; gap:8px }
.au-row { display:grid; grid-template-columns:80px 260px 1fr; gap:12px;
          align-items:baseline; padding:9px 12px; border-radius:8px;
          background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.05) }
.au-status { font:600 11px 'Space Grotesk'; letter-spacing:.06em; text-transform:uppercase }
.au-tool { font:500 12px ui-monospace,Menlo,monospace; color:rgba(232,233,240,.65) }
.au-reason { font:400 12.5px/1.5 'Space Grotesk'; color:rgba(232,233,240,.5) }
.dl-body { padding:0 32px 20px; display:flex; flex-wrap:wrap; gap:8px }
.dl { padding:6px 11px; border-radius:100px; background:rgba(255,255,255,.03);
      border:1px solid; font:500 12px ui-monospace,Menlo,monospace; white-space:nowrap }
.flags { padding:14px 32px; font:400 13px 'Space Grotesk'; color:rgba(232,233,240,.6) }
"""


def _verdict(report: dict) -> tuple[str, str, str, str]:
    knocked = report.get("knocked_out")
    score = report.get("score")
    if knocked:
        return "Knocked out", "—", escape(str(knocked)), "bad"
    if score is not None:
        sub = ("needs review" if report.get("needs_review")
               else escape(str(report.get("scoring_version") or "")))
        return "Score", f"{score:.2f}", sub, "good"
    if report.get("needs_review"):
        return "Needs review", "· ·", "score withheld", "warn"
    reasons = report.get("reasons") or [str(report.get("session_state", ""))]
    return "Not scored", "—", escape(str(reasons[0])), "mute"


def _chip(name: str, f: dict, knocked: str | None) -> str:
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
    return (
        f'<div class="chip" style="background:{bg};border-color:{border}">'
        f'<span class="chip-label" style="color:{label_col}">{label}</span>'
        f'<span class="chip-val" style="color:{val_col}">{escape(name)} = '
        f'{escape(str(f.get("value")))}</span>'
        f'<span class="chip-status" style="color:{status_col}">{status}</span></div>')


def _timeline(report: dict) -> str:
    knocked = report.get("knocked_out")
    by_utterance: dict[str, list] = {}
    for name, f in report.get("fields", {}).items():
        if f.get("utterance_id"):
            by_utterance.setdefault(str(f["utterance_id"]), []).append((name, f))
    rows = []
    for entry in report.get("conversation", []):
        agent = entry.get("role") == "agent"
        who_col = "#a3aefb" if agent else "rgba(232,233,240,.45)"
        chips = "".join(_chip(n, f, knocked) for n, f in
                        by_utterance.get(str(entry.get("utterance_id", "")), []))
        dot = "#7c8cf8" if chips else "#3d4361"
        glow = "0 0 12px rgba(124,140,248,.5)" if chips else "none"
        text_col = "rgba(232,233,240,.9)" if chips else "rgba(232,233,240,.7)"
        interrupted = ('<span class="intr">⏹ interrupted</span>'
                       if entry.get("interrupted") else "")
        rows.append(
            f'<div class="tl-row"><span class="tl-dot" style="background:{dot};'
            f'box-shadow:{glow}"></span>'
            f'<div class="tl-who" style="color:{who_col}">'
            f'{"Asandia" if agent else "Candidate"} {interrupted}</div>'
            f'<div class="tl-text" style="color:{text_col}">'
            f'{escape(str(entry.get("text", "")))}</div>{chips}</div>')
    return "".join(rows)


def _breakdown(report: dict, big: str) -> str:
    subs = report.get("subscores", {})
    if not subs or report.get("score") is None:
        return ""
    max_weight = max((s.get("weight", 0) for s in subs.values()), default=1) or 1
    rows = []
    for name, s in subs.items():
        pct = round(s.get("weighted", 0) / max_weight * 100)
        math = (f'{s.get("subscore", 0):.2f} × {s.get("weight", 0):.2f} '
                f'= {s.get("weighted", 0):.3f}')
        rows.append(
            f'<div class="bk-row"><span class="bk-name">{escape(name)}</span>'
            f'<div class="bk-track"><div class="bk-fill" style="width:{pct}%"></div>'
            f'</div><span class="bk-math">{math}</span></div>')
    return (f'<div class="card pad"><div class="card-head">'
            f'<span class="label">Score breakdown</span>'
            f'<span class="sigma">Σ {big}</span></div>{"".join(rows)}</div>')


def _audit(report: dict) -> str:
    rows = []
    for a in report.get("tool_ledger", []):
        applied = a.get("applied")
        col = "#6ee787" if applied else "#f28488"
        args = a.get("arguments") or {}
        tool = f'{escape(str(a.get("name")))}({escape(str(args.get("field", "")))})'
        reason = escape(str(a.get("reason") or
                            f'turn {a.get("turn_id")} · generation {a.get("generation_id")}'))
        rows.append(
            f'<div class="au-row"><span class="au-status" style="color:{col}">'
            f'{"Applied" if applied else "Rejected"}</span>'
            f'<span class="au-tool">{tool}</span>'
            f'<span class="au-reason">{reason}</span></div>')
    if not rows:
        return ""
    return (f'<details class="card"><summary><span class="label">Audit trail · '
            f'{len(rows)} tool calls</span></summary>'
            f'<div class="au-body">{"".join(rows)}</div></details>')


def _quality(report: dict) -> str:
    delays = report.get("endpoint_delays_ms", [])
    if not delays:
        return ""
    chips = "".join(
        f'<span class="dl" style="color:'
        f'{"#e5b96d" if ms > 700 else "rgba(232,233,240,.6)"};border-color:'
        f'{"rgba(229,168,72,.4)" if ms > 700 else "rgba(255,255,255,.08)"}">'
        f'T{i + 1} · {ms} ms</span>'
        for i, ms in enumerate(delays))
    return ('<details class="card"><summary><span class="label">Call quality · '
            'per-turn endpoint delay</span></summary>'
            f'<div class="dl-body">{chips}</div></details>')


def _unanchored(report: dict) -> str:
    """Evidence that could not be anchored to a conversation utterance must
    still be visible (quarantined goldens, missing ids): lossless fallback."""
    anchored = {str(e.get("utterance_id")) for e in report.get("conversation", [])
                if e.get("utterance_id")}
    knocked = report.get("knocked_out")
    leftovers = [(n, f) for n, f in report.get("fields", {}).items()
                 if str(f.get("utterance_id")) not in anchored]
    if not leftovers:
        return ""
    chips = "".join(_chip(n, f, knocked) for n, f in leftovers)
    return (f'<div class="card pad"><span class="label">Evidence without a '
            f'transcript anchor</span>{chips}</div>')


def render_report_html(report: dict) -> str:
    verdict, big, sub, tone = _verdict(report)
    tile_bg, tile_border, tile_label, sub_col = _TONES[tone]
    fields = report.get("fields", {})
    verified_count = sum(1 for f in fields.values() if f.get("verified"))
    delays = report.get("endpoint_delays_ms", [])
    median_delay = sorted(delays)[len(delays) // 2] if delays else 0
    ko_count = 1 if report.get("knocked_out") else 0
    state = str(report.get("session_state", ""))
    state_ok = state == "completed"
    state_col = "#6ee787" if state_ok else ("#f28488" if state == "failed" else "#e5b96d")
    state_bg = "rgba(110,231,135,.1)" if state_ok else "rgba(229,168,72,.08)"
    state_border = "rgba(110,231,135,.3)" if state_ok else "rgba(229,168,72,.3)"
    call_id = escape(str(report.get("call_id", "")))
    reasons = "; ".join(str(r) for r in (report.get("reasons") or []))
    flags = (f'<div class="card flags"><span class="label">Flags</span> '
             f'{escape(reasons)}</div>' if reasons else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Call report {call_id[:12]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{_CSS}</style></head><body><div class="wrap"><div class="col">
<div class="grid">
  <div class="score-tile" style="background:{tile_bg};border-color:{tile_border}">
    <span class="label" style="color:{tile_label}">{verdict}</span>
    <span class="score-big">{big}</span>
    <span style="font:500 12px 'Space Grotesk';color:{sub_col}">{sub}</span>
  </div>
  <div class="card id-card">
    <div style="display:flex;align-items:center;gap:14px"><div class="mark"></div>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span style="font:600 20px 'Space Grotesk';color:#f2f3fa">Candidate — ICU Registered Nurse</span>
        <span style="font:400 12.5px 'Space Grotesk';color:rgba(232,233,240,.45)">call {call_id[:12]} · {escape(str(report.get("created_at", "")))} · {report.get("turn_count", 0)} turns</span>
      </div></div>
    <span class="state-chip" style="background:{state_bg};border-color:{state_border};color:{state_col}">{escape(state)}</span>
  </div>
  <div class="card stat"><span class="label">Verified fields</span>
    <b>{verified_count}<small> / {len(fields) or "—"}</small></b></div>
  <div class="card stat"><span class="label">Knockouts</span>
    <b style="color:{"#f28488" if ko_count else "#6ee787"}">{ko_count}</b></div>
  <div class="card stat"><span class="label">Median delay</span>
    <b>{median_delay}<small> ms</small></b></div>
</div>
{flags}
<div class="card" style="padding:28px 32px">
  <div class="label" style="padding-bottom:22px">Call replay — every extraction anchored to the moment it was said</div>
  <div class="tl-wrap"><div class="tl-line"></div>{_timeline(report)}</div>
</div>
{_unanchored(report)}
{_breakdown(report, big)}
{_audit(report)}
{_quality(report)}
</div></div></body></html>"""
