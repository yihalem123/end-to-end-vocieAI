"""Silero VAD (ONNX) + hysteresis gate + frame-to-window buffering. Phase 2.

## How this works
Four layers, separable so each is testable alone:
- SileroRuntime owns only the immutable, process-shareable ONNX session.
- SileroVad owns one call's state. Silero v5 is stateful twice over: alongside each
  512-sample 16 kHz window (32 ms) it takes and returns a recurrent state tensor,
  AND each window must be prepended with the last 64 samples of the PREVIOUS
  window (the "context"), so the real model input is 576 samples. Omit the context
  and the model scores real speech near 0.0 — found the hard way in Phase 2, now
  pinned by a regression test on real audio. One inference is ~0.1-0.5 ms on CPU.
- VadGate turns raw per-window speech probabilities into debounced start/stop
  events via hysteresis: `start_windows` consecutive windows >= start_prob to
  start, `stop_windows` consecutive windows <= stop_prob to stop. The asymmetric
  thresholds (start 0.5 / stop 0.35) prevent flicker: mid-band probabilities keep
  the current state and reset the opposing counter.
- VadStream adapts our 320-sample (20 ms) wire frames to the model's 512-sample
  windows with a carry-over buffer, and feeds the gate.
The endpointer (endpoint.py) owns silence *timing*; the gate only answers "is
speech present" quickly (stop after ~3 windows ≈ 96 ms of silence).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime

WINDOW_SAMPLES = 512  # Silero v5 requires exactly 512 samples at 16 kHz (32 ms)
CONTEXT_SAMPLES = 64  # v5 prepends this much of the previous window (16 kHz)
SAMPLE_RATE = 16000
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"


@dataclass(frozen=True)
class VadEvent:
    kind: str  # "start" | "stop"
    t: float   # timestamp of the window that triggered the transition


class SileroRuntime:
    """Immutable model resources that may be shared by concurrent calls."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # tiny model; threading overhead only hurts
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def infer(self, samples: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        out, next_state = self._session.run(
            ["output", "stateN"],
            {
                "input": samples.reshape(1, -1),
                "state": state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        return float(out[0, 0]), next_state


class SileroVad:
    """Per-call recurrent/context state over a shareable SileroRuntime."""

    def __init__(self, runtime: SileroRuntime | None = None) -> None:
        self._runtime = runtime if runtime is not None else SileroRuntime()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def reset(self) -> None:
        self._state[:] = 0.0
        self._context[:] = 0.0

    def prob(self, chunk: np.ndarray) -> float:
        """Speech probability for one float32 window in [-1, 1], shape (512,)."""
        with_context = np.concatenate([self._context, chunk])
        probability, self._state = self._runtime.infer(with_context, self._state)
        self._context = chunk[-CONTEXT_SAMPLES:]
        return probability


class VadGate:
    def __init__(
        self,
        start_prob: float = 0.5,
        stop_prob: float = 0.35,
        start_windows: int = 2,
        stop_windows: int = 3,
    ) -> None:
        self.start_prob = start_prob
        self.stop_prob = stop_prob
        self.start_windows = start_windows
        self.stop_windows = stop_windows
        self.speaking = False
        self._run = 0  # consecutive windows toward the pending transition

    def update(self, prob: float, t: float) -> str | None:
        if not self.speaking:
            if prob >= self.start_prob:
                self._run += 1
                if self._run >= self.start_windows:
                    self.speaking = True
                    self._run = 0
                    return "start"
            else:
                self._run = 0
        else:
            if prob <= self.stop_prob:
                self._run += 1
                if self._run >= self.stop_windows:
                    self.speaking = False
                    self._run = 0
                    return "stop"
            else:
                self._run = 0  # any speech-ish window resets the silence run
        return None


class VadStream:
    def __init__(
        self,
        vad: SileroVad | None = None,
        start_prob: float = 0.5,
        stop_prob: float = 0.35,
        start_windows: int = 2,
        stop_windows: int = 3,
    ) -> None:
        self._vad = vad if vad is not None else SileroVad()
        self._gate = VadGate(start_prob, stop_prob, start_windows, stop_windows)
        self._carry = np.empty(0, dtype=np.float32)

    @property
    def speaking(self) -> bool:
        return self._gate.speaking

    def feed(self, frame: bytes, t: float) -> list[VadEvent]:
        """Consume one PCM16 frame; return any gate transitions it caused."""
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 0x8000
        self._carry = np.concatenate([self._carry, samples])
        events: list[VadEvent] = []
        while len(self._carry) >= WINDOW_SAMPLES:
            window = self._carry[:WINDOW_SAMPLES]
            self._carry = self._carry[WINDOW_SAMPLES:]
            transition = self._gate.update(self._vad.prob(window), t)
            if transition is not None:
                events.append(VadEvent(kind=transition, t=t))
        return events
