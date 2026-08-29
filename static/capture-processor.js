/* Capture worklet: mic samples -> 16 kHz PCM16, 640-byte / 20 ms frames.

## How this works
This class runs on the browser's REALTIME AUDIO THREAD, not the main JS thread —
that's the whole point of AudioWorklet: process() is called every 128 samples
(~2.7 ms at 48 kHz) with a hard deadline, immune to main-thread jank (GC, layout,
our WebSocket code). The audio thread and main thread talk only via this.port
messages, so no shared state and no locks.

The mic arrives as Float32 samples in [-1, 1] at the AudioContext rate (usually
48000 — `sampleRate` here is a worklet global). The server wants 16 kHz PCM16.
We resample by walking a fractional read cursor through the input at a stride of
inRate/16000 (3.0 at 48 kHz) and linearly interpolating between the two samples
around the cursor — cheap, explainable, and fine for speech. Each output sample is
clamped and scaled to a signed 16-bit int. Every 320 samples (20 ms) we post the
frame's ArrayBuffer to the main thread as a TRANSFER (zero-copy handoff: the
buffer moves, it isn't cloned), and start a fresh frame.
*/

const OUT_RATE = 16000;
const FRAME_SAMPLES = 320; // 20 ms at 16 kHz = 640 bytes of PCM16

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stride = sampleRate / OUT_RATE; // input samples per output sample
    // Fixed scratch ring: avoid allocating/copying a Float32Array every audio
    // render quantum (~375 allocations/sec at 48 kHz).
    this.pending = new Float32Array(2048);
    this.pendingLength = 0;
    this.cursor = 0.0;                   // fractional read position into `pending`
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.fill = 0;                       // samples written into `frame` so far
  }

  process(inputs) {
    const input = inputs[0][0]; // mono: first channel of first input
    if (!input) return true;    // mic not delivering yet; keep processor alive

    this.pending.set(input, this.pendingLength);
    this.pendingLength += input.length;

    // Consume while we still have the sample *after* the cursor (needed to
    // interpolate). Anything unconsumed carries over to the next process() call,
    // so no samples are ever dropped at block boundaries.
    let pos = this.cursor;
    while (pos + 1 < this.pendingLength) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const sample = this.pending[i] * (1 - frac) +
                     this.pending[i + 1] * frac; // linear interp
      const clamped = Math.max(-1, Math.min(1, sample));
      this.frame[this.fill++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.fill === FRAME_SAMPLES) {
        const out = this.frame;
        this.port.postMessage(out.buffer, [out.buffer]); // transfer, not copy
        this.frame = new Int16Array(FRAME_SAMPLES);
        this.fill = 0;
      }
      pos += this.stride;
    }

    // Keep the unconsumed tail; cursor becomes fractional offset into it.
    const consumed = Math.floor(pos);
    this.pending.copyWithin(0, consumed, this.pendingLength);
    this.pendingLength -= consumed;
    this.cursor = pos - consumed;
    return true; // false would permanently kill this processor
  }
}

registerProcessor("capture-processor", CaptureProcessor);
