/* Playback worklet: queued 16 kHz PCM16 frames -> speaker output.

## How this works
The mirror of capture-processor: also on the realtime audio thread, but process()
FILLS `outputs` instead of reading `inputs`. The main thread posts each received
640-byte frame to our port; we keep them in a FIFO queue and pull samples out at a
fractional stride of 16000/contextRate (⅓ at 48 kHz — i.e. we upsample by reading
the same source sample for ~3 output ticks, nearest-neighbor; audible quality is
fine for speech, and linear interp is a noted refinement).

Two behaviors matter for later phases:
- UNDERRUN → SILENCE: if the queue is empty we emit zeros rather than stalling.
  Network jitter produces brief gaps, never crashes.
- "clear" MESSAGE → instant flush: drop the whole queue mid-frame. This is the
  barge-in mechanism (Phase 3): the user starts talking, the server says clear,
  and audio stops within one 128-sample block (~2.7 ms) because the very next
  process() call finds nothing to play. This is why playback is a worklet queue
  and not scheduled AudioBufferSources — one flush point, owned by us.
The queue is unbounded here because the server paces TTS frames (Phase 3); the
echo test sends at mic rate, which playback drains at the same rate.
*/

const IN_RATE = 16000;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];   // Int16Array frames, FIFO
    this.cursor = 0.0; // fractional read position within queue[0]
    this.stride = IN_RATE / sampleRate; // source samples per output tick (<1 = upsample)
    this.port.onmessage = (e) => {
      if (e.data === "clear") {
        this.queue.length = 0;
        this.cursor = 0.0;
      } else {
        this.queue.push(new Int16Array(e.data));
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      const frame = this.queue[0];
      if (!frame) {
        out[i] = 0; // underrun: silence, not a stall
        continue;
      }
      out[i] = frame[Math.floor(this.cursor)] / 0x8000; // int16 -> float
      this.cursor += this.stride;
      if (this.cursor >= frame.length) {
        this.queue.shift();
        this.cursor -= frame.length;
      }
    }
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
