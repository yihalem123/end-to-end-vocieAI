"""VAD gate hysteresis, frame windowing, and a real-model smoke test."""
import numpy as np

from server.realtime.vad import SileroVad, VadGate, VadStream, WINDOW_SAMPLES


class FakeVad:
    """Stands in for SileroVad: returns queued probabilities, records chunks."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.chunks: list[np.ndarray] = []

    def prob(self, chunk: np.ndarray) -> float:
        self.chunks.append(chunk)
        return self.probs.pop(0)


# --- VadGate hysteresis (pure logic, no model) ---

def test_gate_starts_after_consecutive_speech_windows() -> None:
    gate = VadGate(start_prob=0.5, stop_prob=0.35, start_windows=2, stop_windows=3)
    assert gate.update(0.9, t=0.00) is None      # one hot window is not enough
    assert gate.update(0.9, t=0.03) == "start"   # second consecutive -> start
    assert gate.speaking


def test_gate_single_spike_does_not_start() -> None:
    gate = VadGate(start_prob=0.5, stop_prob=0.35, start_windows=2, stop_windows=3)
    assert gate.update(0.9, t=0.00) is None
    assert gate.update(0.1, t=0.03) is None      # dip resets the onset counter
    assert gate.update(0.9, t=0.06) is None
    assert not gate.speaking


def test_gate_stops_only_after_consecutive_silence() -> None:
    gate = VadGate(start_prob=0.5, stop_prob=0.35, start_windows=1, stop_windows=3)
    assert gate.update(0.9, t=0.00) == "start"
    assert gate.update(0.1, t=0.03) is None      # 1 quiet window
    assert gate.update(0.1, t=0.06) is None      # 2 quiet windows
    assert gate.update(0.1, t=0.09) == "stop"    # 3rd -> stop
    assert not gate.speaking


def test_gate_mid_band_resets_stop_counter() -> None:
    # Probs between stop_prob and start_prob mean "still speech-ish": they must
    # reset the silence run, or trailing murmurs would end turns prematurely.
    gate = VadGate(start_prob=0.5, stop_prob=0.35, start_windows=1, stop_windows=2)
    assert gate.update(0.9, t=0.00) == "start"
    assert gate.update(0.1, t=0.03) is None
    assert gate.update(0.40, t=0.06) is None     # mid-band: resets counter
    assert gate.update(0.1, t=0.09) is None
    assert gate.update(0.1, t=0.12) == "stop"


# --- VadStream windowing ---

def test_stream_buffers_frames_into_model_windows() -> None:
    # 320-sample frames must be regrouped into 512-sample model windows:
    # 8 frames = 2560 samples = exactly 5 windows.
    fake = FakeVad(probs=[0.0] * 5)
    stream = VadStream(vad=fake, start_windows=1, stop_windows=1)
    frame = np.zeros(320, dtype=np.int16).tobytes()
    for i in range(8):
        stream.feed(frame, t=i * 0.02)
    assert len(fake.chunks) == 5
    assert all(len(c) == WINDOW_SAMPLES for c in fake.chunks)


def test_stream_emits_gate_events() -> None:
    fake = FakeVad(probs=[0.9, 0.9, 0.1, 0.1])
    stream = VadStream(vad=fake, start_windows=2, stop_windows=2)
    frame = np.zeros(WINDOW_SAMPLES, dtype=np.int16).tobytes()  # 1 window per feed
    events = []
    for i in range(4):
        events += stream.feed(frame, t=i * 0.032)
    kinds = [e.kind for e in events]
    assert kinds == ["start", "stop"]


# --- Real model tests ---

def test_real_speech_triggers_gate_events() -> None:
    # Regression: without the 64-sample context the v5 model expects prepended
    # to each window, real speech scores ~0.0 and the gate never fires (found
    # live in Phase 2: transcript flowed but no VAD events, so no turns).
    # tests/assets/speech.wav: 16 kHz mono PCM16, ~6 s of synthesized speech.
    import wave
    from pathlib import Path

    with wave.open(str(Path(__file__).parent / "assets" / "speech.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    stream = VadStream()
    events = []
    for off in range(0, len(pcm) - 639, 640):
        events += stream.feed(pcm[off : off + 640], t=off / 32000)
    kinds = [e.kind for e in events]
    assert "start" in kinds, f"no speech detected in speech audio: {kinds}"
    assert "stop" in kinds, f"speech never ended: {kinds}"


def test_silero_model_loads_and_scores_silence_low() -> None:
    vad = SileroVad()
    silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    p = vad.prob(silence)
    assert 0.0 <= p <= 1.0
    assert p < 0.3  # digital silence must not look like speech
