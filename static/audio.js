/* Main-thread audio glue: mic -> capture worklet -> WS -> server pipeline.

## How this works
Phase 2: the server no longer echoes. Upstream is unchanged — the capture worklet
posts 640-byte PCM16 frames and we ws.send() them. Downstream is now two kinds of
message, split on type:
  - JSON text: pipeline events {vad, partial, final, turn} driving the UI.
    Partials OVERWRITE one gray line (each supersedes the last); finals APPEND
    permanently; a turn event closes out the utterance with its endpoint_delay —
    the number this phase exists to measure.
  - binary ArrayBuffer: audio for the playback worklet (nothing sends it in
    Phase 2; the path stays live because Phase 3 TTS uses it).
Still true from Phase 1: AudioContext needs a user gesture; getUserMedia needs
localhost/https; binaryType "arraybuffer" avoids per-frame Blob reads.
*/

const els = {};
let ctx = null, ws = null, stream = null, running = false;
let playbackNode = null;
let sentCount = 0;

function setStatus(text) { els.status.textContent = text; }

function handleEvent(ev) {
  switch (ev.type) {
    case "vad":
      els.vad.textContent = ev.state === "speech" ? "● speech" : "○ silence";
      els.vad.className = ev.state;
      break;
    case "partial":
      els.partial.textContent = ev.text;
      break;
    case "final": {
      els.partial.textContent = "";
      const span = document.createElement("span");
      span.textContent = ev.text + " ";
      els.finals.appendChild(span);
      break;
    }
    case "turn": {
      const li = document.createElement("li");
      li.textContent = `${ev.endpoint_delay_ms} ms (${ev.reason}) — "${ev.transcript}"`;
      els.turns.prepend(li);
      els.finals.textContent = "";  // turn committed; clear the working line
      addLine("you", ev.transcript);
      break;
    }
    case "agent":
      addLine("agent", ev.text + (ev.interrupted ? " ⏹ (interrupted)" : ""));
      break;
    case "you":  // echo of a chat (text-mode) message
      addLine("you", ev.text);
      break;
    case "clear":
      // Barge-in: flush the playback queue NOW; the worklet reports how many
      // samples were actually heard and we relay that to the server.
      playbackNode?.port.postMessage("clear");
      break;
    case "error":
      setStatus(`server error: ${ev.message}`);
      break;
  }
}

function addLine(who, text) {
  const div = document.createElement("div");
  div.className = `line ${who}`;
  div.textContent = `${who === "agent" ? "🤖" : "🧑"} ${text}`;
  els.convo.appendChild(div);
  els.convo.scrollTop = els.convo.scrollHeight;
}

async function refreshMetrics() {
  try {
    const snap = await fetch("/metrics").then((r) => r.json());
    const rows = Object.entries(snap.stages).map(([stage, s]) =>
      `<tr><td>${stage}</td><td>${s.p50.toFixed(0)}</td><td>${s.p95.toFixed(0)}</td><td>${s.count}</td></tr>`);
    els.metrics.innerHTML =
      `<tr><th>stage</th><th>p50 ms</th><th>p95 ms</th><th>n</th></tr>` + rows.join("");
  } catch { /* server restarting; try again next tick */ }
}

function connectWs() {
  // Shared by voice mode and text mode: one socket, JSON events either way.
  // The mode is fixed at connect time: flux = Deepgram model end-of-turn,
  // otherwise the custom VAD+endpointer stack. Metrics are tagged per mode.
  const mode = els.fluxMode.checked ? "flux" : "custom";
  ws = new WebSocket(`ws://${location.host}/ws/call?mode=${mode}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    if (typeof e.data === "string") handleEvent(JSON.parse(e.data));
    else if (playbackNode) playbackNode.port.postMessage(e.data, [e.data]);
  };
  ws.onclose = () => { if (running) stop("server closed"); ws = null; };
  ws.onerror = () => setStatus("websocket error (is the server running?)");
  return ws;
}

function sendChat() {
  const text = els.chatText.value.trim();
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWs();
    ws.onopen = () => { setStatus("text mode"); ws.send(JSON.stringify({ type: "chat", text })); };
  } else {
    ws.send(JSON.stringify({ type: "chat", text }));
  }
  els.chatText.value = "";
}

async function start() {
  setStatus("requesting mic…");
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
    if (e.data?.type === "cleared" && ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cleared", played_samples: e.data.played }));
    }
  };

  // Reuse a socket opened by text mode, or open a fresh one.
  if (!ws || ws.readyState !== WebSocket.OPEN) connectWs();
  const attachCapture = () => {
    setStatus(`live — context ${ctx.sampleRate} Hz`);
    capture.port.onmessage = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(e.data);
      els.sent.textContent = ++sentCount;
    };
  };
  if (ws.readyState === WebSocket.OPEN) attachCapture();
  else ws.onopen = attachCapture;

  running = true;
  els.fluxMode.disabled = true;  // mode is per-call; toggle applies on next Start
  els.btn.textContent = "Stop";
}

function stop(reason) {
  running = false;
  ws?.close();
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  ws = null; ctx = null; stream = null;
  els.btn.textContent = "Start";
  els.fluxMode.disabled = false;
  setStatus(reason || "stopped");
  pollForReport(6);  // post-call extraction takes a few seconds
}

async function pollForReport(attemptsLeft) {
  if (attemptsLeft <= 0) return;
  try {
    const calls = await fetch("/calls").then((r) => r.json());
    if (calls.length && calls[0].call_id !== pollForReport.shown) {
      pollForReport.shown = calls[0].call_id;
      const div = document.createElement("div");
      div.className = "line agent";
      const a = document.createElement("a");
      a.href = `/report/${calls[0].call_id}/view`;
      a.target = "_blank";
      a.textContent = `📋 call report ready (score ${calls[0].score ?? "?"})`;
      div.appendChild(a);
      els.convo.appendChild(div);
      els.convo.scrollTop = els.convo.scrollHeight;
      return;
    }
  } catch { /* server briefly busy; retry */ }
  setTimeout(() => pollForReport(attemptsLeft - 1), 2500);
}

window.addEventListener("DOMContentLoaded", () => {
  for (const id of ["btn", "status", "sent", "vad", "partial", "finals", "turns",
                    "convo", "metrics", "chatText", "chatSend", "fluxMode"]) {
    els[id] = document.getElementById(id);
  }
  els.btn.addEventListener("click", () => {
    if (running) stop();
    else start().catch((err) => setStatus(`failed: ${err.message}`));
  });
  els.chatSend.addEventListener("click", sendChat);
  els.chatText.addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
  setInterval(refreshMetrics, 2000);
});
