/* Main-thread audio glue: mic -> capture worklet -> WS -> server pipeline.

## How this works
The wire protocol is unchanged from the hardened build: upstream 640-byte PCM16
frames; downstream JSON events plus generation-prefixed binary audio (8-byte id
+ one frame) that must be decoded and acked (cleared / playback_drained /
playback_overflow) with the same generation id. This file adds only the design
console on top: an orb/equalizer stage whose state derives from real signals
(VAD events for the caller, recent binary-frame arrivals for the agent), a
session timer, mic mute via track.enabled, transcript cards, and a latency
panel fed by /metrics/{call_id} p50s (mode-prefixed for the Flux A/B). Session
status text comes from server session_state events, never invented locally.
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

function handleEvent(ev) {
  switch (ev.type) {
    case "session":
      callId = ev.call_id;
      sessionState = ev.state;
      renderStatus();
      break;
    case "session_state":
      sessionState = ev.state;
      renderStatus();
      break;
    case "vad":
      callerSpeaking = ev.state === "speech";
      renderStage();
      break;
    case "partial":
      els.partialText.textContent = ev.text;
      els.partialCard.hidden = !ev.text;
      scrollConvo();
      break;
    case "final":
      els.partialText.textContent = "";
      els.partialCard.hidden = true;
      break;
    case "turn":
      els.partialCard.hidden = true;
      addLine("you", ev.transcript, `${ev.endpoint_delay_ms} ms · ${ev.reason}`);
      break;
    case "agent":
      addLine("agent", ev.text + (ev.interrupted ? " ⏹" : ""),
              ev.interrupted ? "interrupted" : "");
      if (!agentInitiated) {
        agentInitiated = true;
        flushPendingChats();
      }
      break;
    case "you":  // echo of a chat (text-mode) message
      addLine("you", ev.text, "typed");
      break;
    case "notice":
      addLine("system", ev.text, "");
      break;
    case "clear":
      playbackNode?.port.postMessage({ type: "clear", generation_id: ev.generation_id });
      break;
    case "audio_end":
      playbackNode?.port.postMessage({ type: "audio_end", generation_id: ev.generation_id });
      break;
    case "error":
      sessionState = "failed";
      addLine("system", ev.message, "error");
      renderStatus();
      break;
  }
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

function connectWs() {
  // Shared by voice mode and text mode: one socket, JSON events either way.
  // The mode is fixed at connect time: flux = Deepgram model end-of-turn,
  // otherwise the custom VAD+endpointer stack. Metrics are tagged per mode.
  activeMode = els.fluxMode.checked ? "flux" : "custom";
  callId = null;
  agentInitiated = false;
  sessionState = null;
  ws = new WebSocket(`ws://${location.host}/ws/call?mode=${activeMode}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    if (typeof e.data === "string") handleEvent(JSON.parse(e.data));
    else if (playbackNode && e.data.byteLength === 648) {
      const view = new DataView(e.data);
      const generationId = Number(view.getBigUint64(0, false));
      const audio = e.data.slice(8);
      lastAgentAudioAt = performance.now();
      playbackNode.port.postMessage(
        { type: "audio", generation_id: generationId, audio }, [audio]);
    }
  };
  ws.onclose = () => {
    const closedCallId = callId;
    if (running) stop("server closed");
    else { ws = null; renderStatus(); startReportPoll(closedCallId); }
  };
  ws.onerror = () => addLine("system", "websocket error (is the server running?)", "error");
  renderStatus();
  return ws;
}

function sendChat() {
  const text = els.chatText.value.trim();
  if (!text) return;
  pendingChats.push(text);
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWs();
    ws.onopen = () => renderStatus();
  } else if (agentInitiated) {
    flushPendingChats();
  }
  els.chatText.value = "";
}

function flushPendingChats() {
  if (!agentInitiated || ws?.readyState !== WebSocket.OPEN) return;
  while (pendingChats.length) {
    ws.send(JSON.stringify({ type: "chat", text: pendingChats.shift() }));
  }
}

async function start() {
  els.status.textContent = "Requesting mic…";
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/capture-processor.js");
  await ctx.audioWorklet.addModule("/playback-processor.js");

  const source = ctx.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(ctx, "capture-processor");
  const playback = new AudioWorkletNode(ctx, "playback-processor");
  playbackNode = playback;
  source.connect(capture);
  playback.connect(ctx.destination);
  playback.port.onmessage = (e) => {
    if (!e.data?.type || ws?.readyState !== WebSocket.OPEN) return;
    if (["cleared", "playback_drained", "playback_overflow"].includes(e.data.type)) {
      ws.send(JSON.stringify({
        type: e.data.type,
        generation_id: e.data.generation_id,
        played_samples: e.data.played_samples,
      }));
    }
  };

  // Reuse a socket opened by text mode, or open a fresh one.
  if (!ws || ws.readyState !== WebSocket.OPEN) connectWs();
  const attachCapture = () => {
    capture.port.onmessage = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(e.data);
    };
    renderStatus();
  };
  if (ws.readyState === WebSocket.OPEN) attachCapture();
  else ws.onopen = attachCapture;

  running = true;
  muted = false;
  timerStart = performance.now();
  els.idleBlock.hidden = true;
  els.incallBlock.hidden = false;
  els.muteBtn.textContent = "Mute";
  els.muteBtn.classList.remove("muted");
  els.fluxToggle.disabled = true;  // mode is per-call; applies on next session
  renderStage();
}

function toggleMute() {
  if (!stream) return;
  muted = !muted;
  stream.getAudioTracks().forEach((t) => { t.enabled = !muted; });
  els.muteBtn.textContent = muted ? "Unmute" : "Mute";
  els.muteBtn.classList.toggle("muted", muted);
  renderStatus();
}

function stop(reason) {
  const finishedCallId = callId;
  running = false;
  muted = false;
  timerStart = null;
  ws?.close();
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  ws = null; ctx = null; stream = null; playbackNode = null;
  els.idleBlock.hidden = false;
  els.incallBlock.hidden = true;
  els.fluxToggle.disabled = false;
  if (reason) addLine("system", reason, "");
  renderStage();
  startReportPoll(finishedCallId);
}

function startReportPoll(sessionCallId) {
  if (!sessionCallId || reportPolls.has(sessionCallId)) return;
  reportPolls.add(sessionCallId);
  pollForReport(sessionCallId, 6);
}

async function pollForReport(sessionCallId, attemptsLeft) {
  if (attemptsLeft <= 0) {
    reportPolls.delete(sessionCallId);
    return;
  }
  try {
    const response = await fetch(`/report/${encodeURIComponent(sessionCallId)}`);
    if (response.ok) {
      const report = await response.json();
      els.convoEmpty.hidden = true;
      const card = document.createElement("div");
      card.className = "card report";
      const head = document.createElement("div");
      head.className = "who";
      head.textContent = "Call report";
      const body = document.createElement("div");
      body.className = "text";
      const a = document.createElement("a");
      a.href = `/report/${encodeURIComponent(sessionCallId)}/view`;
      a.target = "_blank";
      const verdict = report.knocked_out ? "knocked out"
        : (report.score == null ? "needs review" : `score ${report.score.toFixed(2)}`);
      a.textContent = `Open report → ${verdict}`;
      body.appendChild(a);
      card.append(head, body);
      els.convo.insertBefore(card, els.partialCard);
      scrollConvo();
      reportPolls.delete(sessionCallId);
      return;
    }
  } catch { /* server briefly busy; retry */ }
  setTimeout(() => pollForReport(sessionCallId, attemptsLeft - 1), 2500);
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
