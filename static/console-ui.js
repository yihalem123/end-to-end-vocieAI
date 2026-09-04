/* Console UI: shared state, rendering, and page init.

## How this works
The console is three classic scripts sharing one global lexical scope, loaded
in this order: console-ui.js (this file: every piece of shared state plus all
rendering), ws.js (the call socket, event routing, report polling), and
audio-graph.js (mic, worklets, playback acknowledgements). No bundler and no
modules, on purpose: the page must stay readable as plain files served with
no-store. State here derives only from real signals - VAD events for the
caller, recent binary-frame arrivals for the agent, session_state events for
the status label - never invented locally.
*/

const els = {};
let ctx = null, ws = null, stream = null, running = false;
let playbackNode = null;
let callId = null;
const reportPolls = new Set();
let agentInitiated = false;
const pendingChats = [];

let activeMode = "custom";
let sessionState = null;
let muted = false;
let lastAgentAudioAt = 0;
let callerSpeaking = false;
let timerStart = null;

const STATE_LABELS = {
  disclosure: "Disclosure", awaiting_consent: "Awaiting consent",
  interviewing: "Interviewing", closing: "Wrapping up",
  post_processing: "Analyzing call…", completed: "Completed",
  consent_refused: "Consent declined", failed: "Failed", cancelled: "Cancelled",
};

function agentSpeaking() { return performance.now() - lastAgentAudioAt < 400; }

function renderStatus() {
  let label, dot;
  if (!ws) { label = "Ready"; dot = "var(--idle)"; }
  else if (muted) { label = "Muted"; dot = "var(--amber)"; }
  else if (agentSpeaking()) { label = "Sarah is speaking"; dot = "var(--green)"; }
  else if (running) { label = "Listening…"; dot = "var(--green)"; }
  else { label = "Text mode"; dot = "var(--green)"; }
  if (sessionState && STATE_LABELS[sessionState] &&
      !(running && (sessionState === "interviewing" || sessionState === "awaiting_consent"))) {
    label = STATE_LABELS[sessionState];
  }
  if (sessionState === "failed") dot = "var(--red)";
  els.status.textContent = label;
  els.statusDot.style.background = dot;
}

function renderStage() {
  const talking = agentSpeaking();
  els.eqBars.className = "eq" + (talking ? " agent" : (callerSpeaking && running && !muted ? " user" : ""));
  els.stage.classList.toggle("live", running || talking);
  renderStatus();
}

function renderTimer() {
  if (timerStart === null) return;
  const secs = Math.floor((performance.now() - timerStart) / 1000);
  els.timer.textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}

function scrollConvo() { els.convo.scrollTop = els.convo.scrollHeight; }

function addLine(who, text, meta) {
  els.convoEmpty.hidden = true;
  const card = document.createElement("div");
  card.className = `card ${who}`;
  const head = document.createElement("div");
  head.className = "who";
  head.textContent = who === "agent" ? "Sarah" : (who === "system" ? "System" : "You");
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    head.appendChild(m);
  }
  const body = document.createElement("div");
  body.className = "text";
  body.textContent = text;
  card.append(head, body);
  els.convo.insertBefore(card, els.partialCard);
  scrollConvo();
}

/* ── latency panel: real per-call p50s, mode-prefixed for the Flux A/B ── */
const LAT_STAGES = [
  { key: "endpoint_delay_ms", label: "Endpointing", max: 800 },
  { key: "llm_ttft_ms", label: "LLM TTFT", max: 1500 },
  { key: "tts_ttfb_ms", label: "TTS TTFB", max: 1500 },
  { key: "turn_latency_ms", label: "Turn p95", max: 2500, pct: "p95" },
];

function buildLatencyCards() {
  els.latGrid.innerHTML = "";
  for (const s of LAT_STAGES) {
    const card = document.createElement("div");
    card.className = "lat";
    card.innerHTML = `<div class="row"><span class="lbl">${s.label}</span>` +
      `<span class="ms" data-ms="${s.key}">—</span></div>` +
      `<div class="track"><div class="fill" data-fill="${s.key}"></div></div>`;
    els.latGrid.appendChild(card);
  }
}

async function refreshMetrics() {
  if (!callId) return;
  try {
    const snap = await fetch(`/metrics/${encodeURIComponent(callId)}`).then((r) => r.json());
    const prefix = activeMode === "flux" ? "flux_" : "";
    for (const s of LAT_STAGES) {
      const stage = snap.stages?.[prefix + s.key];
      const value = stage ? stage[s.pct || "p50"] : null;
      const msEl = els.latGrid.querySelector(`[data-ms="${s.key}"]`);
      const fillEl = els.latGrid.querySelector(`[data-fill="${s.key}"]`);
      if (value == null) { msEl.textContent = "—"; fillEl.style.width = "0"; continue; }
      const pct = Math.min(100, Math.round(value / s.max * 100));
      msEl.textContent = `${Math.round(value)} ms`;
      fillEl.style.width = `${pct}%`;
      fillEl.classList.toggle("hot", pct > 75);
    }
    const turn = snap.stages?.[prefix + "turn_latency_ms"];
    els.e2e.textContent = turn ? `e2e ${Math.round(turn.p50)} ms` : "e2e —";
  } catch { /* server restarting; try again next tick */ }
}

function applyTheme(theme) {
  // Light is the main theme; "dark" rides on one attribute + the token table.
  document.documentElement.dataset.theme = theme;
  els.themeToggle.classList.toggle("on", theme === "dark");
  try { localStorage.setItem("theme", theme); } catch { /* private mode */ }
}

window.addEventListener("DOMContentLoaded", () => {
  for (const id of ["btn", "endBtn", "muteBtn", "status", "statusDot", "stage",
                    "eqBars", "idleBlock", "incallBlock", "timer", "convo",
                    "convoEmpty", "chatText", "chatSend", "fluxMode",
                    "fluxToggle", "metricsToggle", "latencyPanel", "latGrid",
                    "e2e", "themeToggle"]) {
    els[id] = document.getElementById(id);
  }
  let saved = "light";
  try { saved = localStorage.getItem("theme") || "light"; } catch { /* ok */ }
  applyTheme(saved);
  els.themeToggle.addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  // The live partial rides in the transcript as a ghost card, updated in place.
  const ghost = document.createElement("div");
  ghost.className = "card ghost";
  ghost.id = "partialCard";
  ghost.hidden = true;
  ghost.innerHTML = `<div class="who">You</div><div class="text" id="partialText"></div>`;
  els.convo.appendChild(ghost);
  els.partialCard = ghost;
  els.partialText = ghost.querySelector("#partialText");

  // Same trick for the agent: her card grows sentence by sentence as the
  // audio for each one starts, then the committed "agent" event replaces it.
  const agentGhost = document.createElement("div");
  agentGhost.className = "card agent ghost";
  agentGhost.id = "agentPartialCard";
  agentGhost.hidden = true;
  agentGhost.innerHTML =
    `<div class="who">Sarah</div><div class="text" id="agentPartialText"></div>`;
  els.convo.appendChild(agentGhost);
  els.agentPartialCard = agentGhost;
  els.agentPartialText = agentGhost.querySelector("#agentPartialText");

  // Ask the server to open a TTS socket now. Connecting measured a median
  // ~2.4 s and is otherwise paid on the greeting — the first thing the caller
  // hears. Best effort: if it fails the call just connects on demand.
  fetch("/prewarm", { method: "POST" }).catch(() => {});

  buildLatencyCards();
  els.btn.addEventListener("click", () =>
    start().catch((err) => addLine("system", `mic failed: ${err.message}`, "error")));
  els.endBtn.addEventListener("click", () => stop());
  els.muteBtn.addEventListener("click", toggleMute);
  els.metricsToggle.addEventListener("click", () => {
    const on = els.metricsToggle.classList.toggle("on");
    els.latencyPanel.hidden = !on;
  });
  els.fluxToggle.addEventListener("click", () => {
    if (els.fluxToggle.disabled) return;
    els.fluxMode.checked = !els.fluxMode.checked;
    els.fluxToggle.classList.toggle("on", els.fluxMode.checked);
  });
  els.chatSend.addEventListener("click", sendChat);
  els.chatText.addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
  setInterval(refreshMetrics, 2000);
  setInterval(() => { renderStage(); renderTimer(); }, 250);
});
