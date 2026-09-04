/* Call socket: JSON events in, audio frames out to the playback worklet.

## How this works
One WebSocket per call, shared by voice and text mode. Downstream frames are
generation-prefixed (8-byte id + one 640-byte frame); everything else is JSON.
handleEvent() is the single switch that turns server events into UI state -
partials and the agent's sentence-by-sentence ghost card update in place,
committed turns and replies become cards. The report poll is keyed to THIS
session's call id, never to a global 'newest call' (a test pins that).
*/

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
    case "agent_partial":
      els.agentPartialText.textContent = ev.text;
      els.agentPartialCard.hidden = !ev.text;
      scrollConvo();
      break;
    case "agent":
      els.agentPartialText.textContent = "";
      els.agentPartialCard.hidden = true;
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

function connectWs() {
  // Shared by voice mode and text mode: one socket, JSON events either way.
  // The mode is fixed at connect time: flux = Deepgram model end-of-turn,
  // otherwise the custom VAD+endpointer stack. Metrics are tagged per mode.
  activeMode = els.fluxMode.checked ? "flux" : "custom";
  callId = null;
  agentInitiated = false;
  sessionState = null;
  // Scheme follows the page: wss behind https (an ngrok/Fly origin), ws on
  // plain localhost. A hard-coded ws:// is blocked as mixed content over https.
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/ws/call?mode=${activeMode}`);
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
