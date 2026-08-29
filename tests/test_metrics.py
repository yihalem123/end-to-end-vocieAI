"""Per-stage latency registry: percentiles, bounded history, snapshot shape."""
from server.metrics import MetricsRegistry, percentile


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert percentile(values, 50) == 25.0  # midpoint of 20 and 30


def test_percentile_single_value() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_registry_snapshot_per_stage() -> None:
    reg = MetricsRegistry()
    for ms in (100, 200, 300):
        reg.record_turn(endpoint_delay_ms=ms, tts_ttfb_ms=ms + 5)
    snap = reg.snapshot()
    assert snap["turns"] == 3
    assert snap["stages"]["endpoint_delay_ms"]["p50"] == 200
    assert snap["stages"]["tts_ttfb_ms"]["p50"] == 205
    assert snap["stages"]["endpoint_delay_ms"]["p95"] > 200


def test_registry_ignores_missing_stages() -> None:
    # Phase 2 turns have no TTS timings; Phase 4 adds llm_ttft. Turns record
    # whatever stages they have, and each stage aggregates independently.
    reg = MetricsRegistry()
    reg.record_turn(endpoint_delay_ms=100)
    reg.record_turn(endpoint_delay_ms=200, tts_ttfb_ms=50)
    snap = reg.snapshot()
    assert snap["stages"]["endpoint_delay_ms"]["count"] == 2
    assert snap["stages"]["tts_ttfb_ms"]["count"] == 1


def test_registry_history_is_bounded() -> None:
    reg = MetricsRegistry(max_turns=10)
    for i in range(50):
        reg.record_turn(endpoint_delay_ms=i)
    snap = reg.snapshot()
    assert snap["turns"] == 50                            # lifetime count keeps counting
    assert snap["stages"]["endpoint_delay_ms"]["count"] == 10  # window is bounded
    assert snap["stages"]["endpoint_delay_ms"]["p50"] == 44.5  # only recent values


def test_registry_filters_metrics_by_stable_call_id() -> None:
    reg = MetricsRegistry()
    reg.record_turn("call-a", endpoint_delay_ms=100)
    reg.record_turn("call-b", endpoint_delay_ms=900)
    snap = reg.snapshot("call-a")
    assert snap["call_id"] == "call-a"
    assert snap["turns"] == 1
    assert snap["stages"]["endpoint_delay_ms"]["p50"] == 100
