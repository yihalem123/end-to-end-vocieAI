"""Browser mic capture worklet, exercised as real JS (see capture_harness.js).

The simulator streams PCM straight over the WebSocket, so it can never see a
broken capture worklet. These tests are the only automated coverage of the
browser's microphone path.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent / "capture_harness.js"
BLOCKS = 200          # render quanta fed per rate
QUANTUM = 128
FRAME_SAMPLES = 320   # 20 ms at 16 kHz


@pytest.fixture(scope="module")
def worklet_runs() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is required to exercise the capture worklet")
    proc = subprocess.run(["node", str(HARNESS)], capture_output=True,
                          text=True, check=True)
    return json.loads(proc.stdout)


@pytest.mark.parametrize("rate", ["48000", "44100", "16000"])
def test_worklet_survives_continuous_audio(worklet_runs, rate) -> None:
    # Live regression: a scratch-ring rewrite let the consumed count exceed the
    # buffered count, so pendingLength went negative and the SECOND process()
    # threw RangeError. Chrome kills a throwing processor: the mic went dead
    # for the whole call and the server logged "0 frames in".
    run = worklet_runs[rate]
    assert run["error"] is None, run["error"]
    assert run["blocks_survived"] == BLOCKS


@pytest.mark.parametrize("rate", ["48000", "44100", "16000"])
def test_worklet_emits_full_20ms_frames_at_the_expected_rate(worklet_runs, rate) -> None:
    run = worklet_runs[rate]
    expected = (BLOCKS * QUANTUM) * (16000 / float(rate)) / FRAME_SAMPLES
    assert run["frames"] == pytest.approx(expected, abs=1)
    assert run["frame_bytes"] == FRAME_SAMPLES * 2  # 640-byte PCM16 frames


@pytest.mark.parametrize("rate", ["48000", "44100", "16000"])
def test_worklet_resamples_signal_not_silence(worklet_runs, rate) -> None:
    # A resampler that reads past its own buffer emits zeros and still looks
    # "alive"; assert the 440 Hz half-scale tone survives the conversion.
    assert worklet_runs[rate]["peak"] > 0.4 * 0x7fff
