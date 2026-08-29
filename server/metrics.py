"""Per-stage latency registry backing /metrics. Phase 3.

## How this works
Every completed turn records whatever stage timings it has (endpoint_delay_ms,
tts_ttfb_ms, first_audio_ms, later llm_ttft_ms) as one dict. The registry keeps a
bounded deque of recent turns — percentiles over the last N are what you want
operationally (recent behavior), and memory stays flat on long runs — plus a
lifetime counter. snapshot() computes p50/p95/count per stage, skipping turns
that lack a stage, so phases can add stages without migrations. Percentile uses
linear interpolation between ranks (same convention as numpy's default): for
small samples it behaves sensibly where nearest-rank would jump around.
"""
from collections import deque


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


class MetricsRegistry:
    def __init__(self, max_turns: int = 500) -> None:
        self._turns: deque[tuple[str | None, dict[str, float]]] = deque(maxlen=max_turns)
        self._lifetime = 0

    def record_turn(self, call_id: str | None = None, **stages: float) -> None:
        self._turns.append((call_id, dict(stages)))
        self._lifetime += 1

    def snapshot(self, call_id: str | None = None) -> dict:
        by_stage: dict[str, list[float]] = {}
        selected = [(owner, turn) for owner, turn in self._turns
                    if call_id is None or owner == call_id]
        for _, turn in selected:
            for stage, value in turn.items():
                by_stage.setdefault(stage, []).append(value)
        return {
            "call_id": call_id,
            "turns": self._lifetime if call_id is None else len(selected),
            "stages": {
                stage: {
                    "count": len(values),
                    "p50": percentile(values, 50),
                    "p95": percentile(values, 95),
                }
                for stage, values in by_stage.items()
            },
        }


registry = MetricsRegistry()  # process-wide; sessions record, /metrics reads
