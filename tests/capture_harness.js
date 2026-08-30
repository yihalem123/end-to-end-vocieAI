/* Drives the REAL static/capture-processor.js under Node.

The mic path is the one stage no Python test and no simulator can reach: the
simulator sends PCM straight over the WebSocket, so a broken capture worklet
passes every live gate while the browser sends nothing (that is exactly how a
regression shipped). This harness stubs the three AudioWorklet globals
(sampleRate, AudioWorkletProcessor, registerProcessor), feeds render quanta,
and reports what the worklet posted. Output: one JSON object on stdout. */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = fs.readFileSync(
  path.join(__dirname, "..", "static", "capture-processor.js"), "utf8");
const QUANTUM = 128; // samples per process() call, fixed by the Web Audio spec

function run(sampleRate, blocks) {
  const posted = [];
  let registered = null;
  const sandbox = {
    sampleRate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (buf) => posted.push(buf) };
      }
    },
    registerProcessor: (_name, cls) => { registered = cls; },
  };
  vm.createContext(sandbox);
  vm.runInContext(SRC, sandbox);

  const proc = new registered();
  const block = new Float32Array(QUANTUM);
  let error = null;
  let survived = 0;
  try {
    for (let i = 0; i < blocks; i++) {
      for (let j = 0; j < QUANTUM; j++) {
        // 440 Hz tone at half scale: real signal, so silence is detectable.
        block[j] = Math.sin(2 * Math.PI * 440 * ((i * QUANTUM + j) / sampleRate)) * 0.5;
      }
      proc.process([[block]]);
      survived = i + 1;
    }
  } catch (e) {
    error = `${e.constructor.name}: ${e.message}`;
  }

  let peak = 0;
  for (const buf of posted) {
    const view = new Int16Array(buf);
    for (const s of view) peak = Math.max(peak, Math.abs(s));
  }
  return {
    error,
    blocks_survived: survived,
    frames: posted.length,
    frame_bytes: posted.length ? posted[0].byteLength : 0,
    peak,
  };
}

const BLOCKS = 200; // ~0.5 s of mic at 48 kHz
console.log(JSON.stringify({
  "48000": run(48000, BLOCKS),
  "44100": run(44100, BLOCKS),
  "16000": run(16000, BLOCKS),
}));
