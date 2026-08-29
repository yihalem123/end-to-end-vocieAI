"""overlap_stream: on-demand filler — spoken only when the engine is slow."""
import asyncio

import pytest

from server.realtime.reply import FILLERS, overlap_stream, wants_filler

PATIENCE = 0.05  # fast test clock


class FakeInterview:
    def __init__(self, fields: dict) -> None:
        self.fields = fields


def test_wants_filler_gating() -> None:
    assert wants_filler(None) is False                     # stub engine: instant
    assert wants_filler(FakeInterview({})) is False        # first exchange: no
    assert wants_filler(FakeInterview({"consent": 1})) is True


def test_fast_engine_skips_the_filler() -> None:
    async def quick():
        yield "Here already."

    async def run() -> list[str]:
        return [s async for s in overlap_stream("Okay.", quick(),
                                                patience_sec=PATIENCE)]

    assert asyncio.run(run()) == ["Here already."]          # no filler needed


def test_slow_engine_gets_covered_by_the_filler() -> None:
    async def slow():
        await asyncio.sleep(PATIENCE * 4)
        yield "Sorry, had to think."

    async def run() -> list[str]:
        return [s async for s in overlap_stream("Okay.", slow(),
                                                patience_sec=PATIENCE)]

    assert asyncio.run(run()) == ["Okay.", "Sorry, had to think."]


def test_engine_runs_concurrently_during_patience_window() -> None:
    started = asyncio.Event()

    async def inner():
        started.set()  # the request must begin during the wait, not after
        await asyncio.sleep(PATIENCE * 4)
        yield "reply"

    async def run() -> None:
        gen = overlap_stream("Okay.", inner(), patience_sec=PATIENCE)
        assert await anext(gen) == "Okay."
        assert started.is_set(), "engine must start before the filler decision"
        assert await anext(gen) == "reply"

    asyncio.run(run())


def test_overlap_stream_propagates_engine_errors() -> None:
    async def broken():
        raise RuntimeError("engine down")
        yield  # pragma: no cover

    async def run() -> None:
        gen = overlap_stream("Okay.", broken(), patience_sec=PATIENCE)
        with pytest.raises(RuntimeError, match="engine down"):
            await anext(gen)

    asyncio.run(run())


def test_fillers_are_short_spoken_lines() -> None:
    assert all(f.endswith(".") and len(f) <= 12 for f in FILLERS)
