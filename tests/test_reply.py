"""overlap_stream: filler yields immediately, engine runs underneath."""
import asyncio

import pytest

from server.realtime.reply import FILLERS, overlap_stream, wants_filler


class FakeInterview:
    def __init__(self, fields: dict) -> None:
        self.fields = fields


def test_wants_filler_gating() -> None:
    assert wants_filler(None) is False                     # stub engine: instant
    assert wants_filler(FakeInterview({})) is False        # first exchange: no
    assert wants_filler(FakeInterview({"consent": 1})) is True


def test_overlap_stream_starts_inner_before_first_is_consumed() -> None:
    started = asyncio.Event()

    async def inner():
        started.set()  # proves the engine request began without being pulled
        yield "reply sentence"

    async def run() -> list[str]:
        out = []
        gen = overlap_stream("Okay.", inner())
        first = await anext(gen)
        await asyncio.sleep(0)  # let the pump task run
        assert started.is_set(), "engine must start concurrently with the filler"
        out.append(first)
        async for s in gen:
            out.append(s)
        return out

    assert asyncio.run(run()) == ["Okay.", "reply sentence"]


def test_overlap_stream_propagates_engine_errors() -> None:
    async def broken():
        raise RuntimeError("engine down")
        yield  # pragma: no cover

    async def run() -> None:
        gen = overlap_stream("Okay.", broken())
        assert await anext(gen) == "Okay."
        with pytest.raises(RuntimeError, match="engine down"):
            await anext(gen)

    asyncio.run(run())


def test_fillers_are_short_spoken_lines() -> None:
    assert all(f.endswith(".") and len(f) <= 12 for f in FILLERS)
