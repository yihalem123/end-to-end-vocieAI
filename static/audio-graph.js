/* Audio graph: mic -> capture worklet -> socket; socket -> playback worklet.

## How this works
getUserMedia with echo cancellation and noise suppression, then two
AudioWorklets on one AudioContext: capture-processor resamples the mic to
16 kHz PCM16 20 ms frames on the realtime thread and posts each frame here to
be sent as binary; playback-processor consumes generation-tagged frames and
reports cleared / playback_started / playback_drained / playback_overflow
with the same generation id, which this file forwards as JSON acks. Mute is
track.enabled, so the graph stays alive and the worklet keeps its state.
*/

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
    if (["cleared", "playback_started", "playback_drained",
         "playback_overflow"].includes(e.data.type)) {
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
