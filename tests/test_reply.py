"""overlap_stream: on-demand filler — spoken only when the engine is slow."""
import asyncio
import time
from types import SimpleNamespace

import pytest

import server.realtime.reply as reply_module
from server.engine.stub import StubEngine
from server.realtime.bargein import BargeInGuard
from server.realtime.endpoint import TurnComplete
from server.realtime.reply import (
    DRAIN_ACK_MARGIN_SEC,
    DRAIN_ACK_MAX_SEC,
    FILLERS,
    drain_wait_seconds,
    overlap_stream,
    wait_for_playback_drain,
    wants_filler,
)
from server.realtime.supervisor import GenerationSupervisor

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


def test_drain_wait_is_expected_remaining_plus_bounded_margin() -> None:
    assert drain_wait_seconds(0.0) == DRAIN_ACK_MARGIN_SEC
    assert drain_wait_seconds(0.25) == DRAIN_ACK_MARGIN_SEC + 0.25
    assert drain_wait_seconds(999.0) == DRAIN_ACK_MAX_SEC


def test_missing_drain_ack_times_out_instead_of_wedging() -> None:
    async def run() -> None:
        drained = asyncio.Event()
        assert not await wait_for_playback_drain(drained, 4, timeout_sec=0.01)

    asyncio.run(run())


def test_drain_ack_completes_without_waiting_for_timeout() -> None:
    async def run() -> None:
        drained = asyncio.Event()
        drained.set()
        assert await wait_for_playback_drain(drained, 4, timeout_sec=1.0)

    asyncio.run(run())


def test_voice_pipeline_finalizes_when_client_never_acks(monkeypatch) -> None:
    monkeypatch.setattr(reply_module, "DRAIN_ACK_MARGIN_SEC", 0.01)
    monkeypatch.setattr(reply_module, "DRAIN_ACK_MAX_SEC", 0.02)

    class FakeSpeaker:
        async def speak(self, *_args):
            return {"_playback_remaining_sec": 0.0}

        def text(self, _generation_id: int) -> str:
            return "Reply completed."

    async def run() -> None:
        sent: list[dict] = []
        finalized = asyncio.Event()

        async def send(event: dict) -> None:
            sent.append(event)
            if event["type"] == "agent":
                finalized.set()

        controller = object.__new__(reply_module.ReplyController)
        controller._send = send
        controller._speaker = FakeSpeaker()
        controller._supervisor = GenerationSupervisor()
        controller._drained = {}
        controller._active_voice_generation = None
        controller._metric_prefix = ""
        controller._call_id = "test-call"
        controller._state = SimpleNamespace(conversation=[])
        controller.guard = BargeInGuard()
        controller.interview = None
        controller._filler_idx = 0
        controller._engine = StubEngine()
        now = time.monotonic()
        turn = TurnComplete("Hello.", 0.1, now, now, "test")

        token = await controller._supervisor.start(
            9, lambda owned: controller._speak_reply(owned, turn))
        controller._drained[token.generation_id] = asyncio.Event()
        controller._active_voice_generation = token.generation_id
        controller.guard.on_agent_audio_start()
        await asyncio.wait_for(finalized.wait(), timeout=0.2)

        assert [event["type"] for event in sent] == ["audio_end", "agent"]
        assert controller._state.conversation[0]["text"] == "Reply completed."
        assert not controller.guard.agent_speaking

    asyncio.run(run())
