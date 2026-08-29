/* Main-thread audio glue: mic -> capture worklet -> WS -> playback worklet.

## How this works
The main thread never touches raw audio samples — it only moves ArrayBuffers
between three parties:
  1. capture-processor posts a 640-byte frame  -> we ws.send() it (binary)
  2. the server echoes it back                 -> ws.onmessage fires
  3. we post the buffer into playback-processor -> it reaches the speakers
RTT is measured with a FIFO of send timestamps: WebSocket messages are ordered,
and the server echoes 1-for-1, so the Nth frame back matches the Nth timestamp —
no sequence numbers needed *for the echo phase*. performance.now() gives
monotonic sub-ms time.

Gotchas worth defending in an interview:
- AudioContext must be created/resumed inside a user gesture (the Start click) —
  browsers block autoplaying audio contexts.
- getUserMedia requires a secure context: https OR localhost. Our localhost dev
  setup is exactly the carve-out.
- echoCancellation stays ON so the mic doesn't re-capture what the speakers play
  (still: use headphones — AEC is tuned for far-end voices, not your own echo).
- ws.binaryType = "arraybuffer": the default is Blob, which would force an async
  read on every frame.
*/

const els = {
  btn: null, status: null, sent: null, recv: null, rtt: null,
};
let ctx = null, ws = null, stream = null, running = false;
let sentCount = 0, recvCount = 0;
const sendTimes = [];   // FIFO of performance.now() per outbound frame
const rttWindow = [];   // last N round-trip times, for a rolling average

function setStatus(text) { els.status.textContent = text; }

async function start() {
  setStatus("requesting mic…");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  ctx = new AudioContext(); // hardware rate (usually 48k); worklets resample
  await ctx.audioWorklet.addModule("/capture-processor.js");
  await ctx.audioWorklet.addModule("/playback-processor.js");

  const source = ctx.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(ctx, "capture-processor");
  const playback = new AudioWorkletNode(ctx, "playback-processor");
  source.connect(capture);          // mic into the capture worklet
  playback.connect(ctx.destination); // playback worklet to the speakers
  // capture is NOT connected to destination — frames leave via port, not audio graph.

  ws = new WebSocket(`ws://${location.host}/ws/call`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus(`live — context ${ctx.sampleRate} Hz, frames 20 ms`);
    capture.port.onmessage = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      sendTimes.push(performance.now());
      ws.send(e.data);
      els.sent.textContent = ++sentCount;
    };
  };

  ws.onmessage = (e) => {
    const t0 = sendTimes.shift();
    if (t0 !== undefined) {
      rttWindow.push(performance.now() - t0);
      if (rttWindow.length > 50) rttWindow.shift();
      const avg = rttWindow.reduce((a, b) => a + b, 0) / rttWindow.length;
      els.rtt.textContent = avg.toFixed(1);
    }
    playback.port.postMessage(e.data, [e.data]); // transfer into the audio thread
    els.recv.textContent = ++recvCount;
  };

  ws.onclose = () => { if (running) stop("server closed"); };
  ws.onerror = () => setStatus("websocket error (is the server running?)");

  running = true;
  els.btn.textContent = "Stop";
}

function stop(reason) {
  running = false;
  ws?.close();
  stream?.getTracks().forEach((t) => t.stop()); // release the mic (tab indicator off)
  ctx?.close();
  ws = null; ctx = null; stream = null;
  sendTimes.length = 0; rttWindow.length = 0;
  els.btn.textContent = "Start";
  setStatus(reason || "stopped");
}

window.addEventListener("DOMContentLoaded", () => {
  els.btn = document.getElementById("toggle");
  els.status = document.getElementById("status");
  els.sent = document.getElementById("sent");
  els.recv = document.getElementById("recv");
  els.rtt = document.getElementById("rtt");
  els.btn.addEventListener("click", () => {
    if (running) stop();
    else start().catch((err) => setStatus(`failed: ${err.message}`));
  });
});
