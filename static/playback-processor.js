/* Generation-aware, bounded playback queue for 16 kHz PCM16 agent audio.

Every audio and control message carries a generation id. Older generations are
ignored, clear is acknowledged with the exact played-sample count, and an end
marker is acknowledged only after its final buffered sample is audible. Queue
overflow fails closed: buffered audio is discarded and that generation is
blocked so a slow browser cannot replay a stale, arbitrarily large backlog.
*/

const IN_RATE = 16000;
const MAX_QUEUE_FRAMES = 75; // 1.5 seconds at 20 ms/frame

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.cursor = 0.0;
    this.stride = IN_RATE / sampleRate;
    this.played = 0;
    this.generationId = 0;
    this.blockedThrough = 0;
    this.endedGeneration = 0;
    this.drainedGeneration = 0;
    this.startedGeneration = 0;
    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  onMessage(message) {
    const generationId = Number(message?.generation_id || 0);
    if (generationId <= 0) return;
    if (message.type === "audio") this.enqueue(generationId, message.audio);
    else if (message.type === "clear") this.clear(generationId);
    else if (message.type === "audio_end") this.end(generationId);
  }

  enqueue(generationId, audio) {
    if (generationId <= this.blockedThrough || generationId < this.generationId) return;
    if (generationId > this.generationId) {
      if (this.queue.length) this.failClosed(this.generationId);
      this.generationId = generationId;
      this.played = 0;
      this.cursor = 0.0;
      this.endedGeneration = 0;
      this.drainedGeneration = 0;
      this.startedGeneration = 0;
    }
    if (!(audio instanceof ArrayBuffer) || audio.byteLength !== 640) return;
    if (this.queue.length >= MAX_QUEUE_FRAMES) {
      this.failClosed(generationId);
      return;
    }
    this.queue.push(new Int16Array(audio));
  }

  clear(generationId) {
    if (generationId < this.generationId || generationId <= this.blockedThrough) return;
    if (generationId > this.generationId) this.generationId = generationId;
    const playedSamples = this.foldAndClear();
    this.blockedThrough = Math.max(this.blockedThrough, generationId);
    this.endedGeneration = 0;
    this.port.postMessage({ type: "cleared", generation_id: generationId, played_samples: playedSamples });
  }

  end(generationId) {
    if (generationId !== this.generationId || generationId <= this.blockedThrough) return;
    this.endedGeneration = generationId;
    this.reportDrainedIfReady();
  }

  failClosed(generationId) {
    const playedSamples = this.foldAndClear();
    this.blockedThrough = Math.max(this.blockedThrough, generationId);
    this.endedGeneration = 0;
    this.port.postMessage({
      type: "playback_overflow", generation_id: generationId, played_samples: playedSamples,
    });
  }

  foldAndClear() {
    const playedSamples = this.played + Math.floor(this.cursor);
    this.queue.length = 0;
    this.cursor = 0.0;
    this.played = playedSamples;
    return playedSamples;
  }

  reportDrainedIfReady() {
    if (this.queue.length || this.endedGeneration !== this.generationId ||
        this.drainedGeneration === this.generationId) return;
    this.drainedGeneration = this.generationId;
    this.port.postMessage({ type: "playback_drained", generation_id: this.generationId });
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      const frame = this.queue[0];
      if (!frame) {
        out[i] = 0;
        continue;
      }
      if (this.startedGeneration !== this.generationId) {
        this.startedGeneration = this.generationId;
        this.port.postMessage({
          type: "playback_started", generation_id: this.generationId,
          played_samples: this.played + Math.floor(this.cursor),
        });
      }
      out[i] = frame[Math.floor(this.cursor)] / 0x8000;
      this.cursor += this.stride;
      if (this.cursor >= frame.length) {
        this.queue.shift();
        this.cursor -= frame.length;
        this.played += frame.length;
      }
    }
    this.reportDrainedIfReady();
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
