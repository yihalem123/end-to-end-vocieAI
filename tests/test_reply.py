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


def test_voice_engine_failure_produces_exactly_one_error_and_no_reply(monkeypatch) -> None:
    """P3.4: a failed generation must fail closed — never a second, conflicting
    reply after audio already went out, and never a silent double-append."""
    class ExplodingSpeaker:
        async def speak(self, *_args):
            raise RuntimeError("provider died mid-reply")

        def text(self, _generation_id: int) -> str:  # pragma: no cover
            return "never"

    async def run() -> None:
        sent: list[dict] = []
        done = asyncio.Event()

        async def send(event: dict) -> None:
            sent.append(event)
            if event["type"] == "error":
                done.set()

        controller = object.__new__(reply_module.ReplyController)
        controller._send = send
        controller._speaker = ExplodingSpeaker()
        controller._supervisor = GenerationSupervisor()
        controller._drained = {}
        controller._active_voice_generation = None
        controller._metric_prefix = ""
        controller._state = SimpleNamespace(conversation=[])
        controller.guard = BargeInGuard()
        controller.interview = None
        controller._filler_idx = 0
        controller._engine = StubEngine()
        now = time.monotonic()
        turn = TurnComplete("Hello.", 0.1, now, now, "test")
        token = await controller._supervisor.start(
            1, lambda owned: controller._speak_reply(owned, turn))
        controller._drained[token.generation_id] = asyncio.Event()
        controller._active_voice_generation = token.generation_id
        controller.guard.on_agent_audio_start()
        await asyncio.wait_for(done.wait(), timeout=0.5)
        await asyncio.sleep(0.05)  # would catch any trailing duplicate reply

        assert [e["type"] for e in sent] == ["error"]
        assert controller._state.conversation == []   # nothing fabricated
        assert not controller.guard.agent_speaking

    asyncio.run(run())


# --- speculative generation: commit-gated audio release ---

class FakeLlm(reply_module.LlmEngine):
    """isinstance-compatible engine double; no network, records transcripts."""
    def __init__(self):  # noqa: super().__init__ deliberately skipped
        self.transcripts = []
        self.last_ttft_ms = None

    async def respond(self, transcript, **_kw):
        self.transcripts.append(transcript)
        yield f"Reply to: {transcript}"


class GatedSpeaker:
    def __init__(self, events):
        self.events = events

    async def speak(self, sentences, anchors, generation_id, is_current, release=None):
        async for _ in sentences:
            pass
        self.events.append("engine-done")
        if release is not None:
            await release.wait()
        self.events.append(("released", anchors["commit_t"]))
        return {"_playback_remaining_sec": 0.0}

    def text(self, generation_id):
        return "Spoken reply."


def _controller(events, engine):
    controller = object.__new__(reply_module.ReplyController)
    controller._send = events["send"]
    controller._speaker = GatedSpeaker(events["log"])
    controller._supervisor = GenerationSupervisor()
    controller._drained = {}
    controller._spec = None
    controller._active_voice_generation = None
    controller._metric_prefix = ""
    controller._call_id = "test-call"
    controller._state = SimpleNamespace(conversation=[])
    controller.guard = BargeInGuard()
    controller.interview = None
    controller._filler_idx = 0
    controller._engine = engine
    return controller


def _spec_env(monkeypatch):
    monkeypatch.setattr(reply_module, "DRAIN_ACK_MARGIN_SEC", 0.01)
    monkeypatch.setattr(reply_module, "DRAIN_ACK_MAX_SEC", 0.02)


def test_speculation_promotes_without_restart_and_gates_audio(monkeypatch) -> None:
    _spec_env(monkeypatch)

    async def run() -> None:
        sent, log = [], []
        done = asyncio.Event()

        async def send(ev):
            sent.append(ev)
            if ev["type"] == "agent":
                done.set()

        controller = _controller({"send": send, "log": log}, FakeLlm())
        now = time.monotonic()
        await controller.speculate("Hello there.", turn_id=5)
        spec_generation = controller._spec["token"].generation_id
        await asyncio.sleep(0.05)  # engine+tts run warm...
        assert "engine-done" in log
        assert not any(isinstance(e, tuple) for e in log)  # ...but NO audio released

        turn = TurnComplete("Hello there.", 0.05, now, now + 0.25, "fast")
        await controller.on_turn(turn, 5)
        await asyncio.wait_for(done.wait(), timeout=0.5)
        released = [e for e in log if isinstance(e, tuple)]
        assert released == [("released", turn.commit_t)]  # anchors from the commit
        assert controller._supervisor._next_generation_id == spec_generation  # no restart
        assert controller._state.conversation[0]["text"] == "Spoken reply."

    asyncio.run(run())


def test_speculation_mismatch_cancels_and_runs_fresh(monkeypatch) -> None:
    _spec_env(monkeypatch)

    async def run() -> None:
        sent, log = [], []
        done = asyncio.Event()

        async def send(ev):
            sent.append(ev)
            if ev["type"] == "agent":
                done.set()

        engine = FakeLlm()
        controller = _controller({"send": send, "log": log}, engine)
        now = time.monotonic()
        await controller.speculate("I think maybe.", turn_id=5)
        await asyncio.sleep(0.02)
        turn = TurnComplete("I think maybe not.", 0.05, now, now + 0.25, "fast")
        await controller.on_turn(turn, 5)
        await asyncio.wait_for(done.wait(), timeout=0.5)
        assert engine.transcripts == ["I think maybe.", "I think maybe not."]
        assert controller._spec is None
        # exactly one release, for the fresh (non-gated) generation
        assert len([e for e in log if isinstance(e, tuple)]) == 1

    asyncio.run(run())


def test_caller_resume_cancels_speculation_silently() -> None:
    async def run() -> None:
        log = []

        async def send(_ev):
            raise AssertionError("nothing should reach the client")

        controller = _controller({"send": send, "log": log}, FakeLlm())
        await controller.speculate("Yes.", turn_id=3)
        await asyncio.sleep(0.02)
        await controller.cancel_speculation()
        assert controller._supervisor.current is None
        assert not any(isinstance(e, tuple) for e in log)  # no audio ever released
        assert controller._state.conversation == []

    asyncio.run(run())


def test_on_script_speaks_and_finalizes(monkeypatch) -> None:
    # Regression: on_script kept the old speak/deliver signature after the
    # anchors refactor and every scripted (disclosure/consent) reply crashed.
    _spec_env(monkeypatch)

    async def run() -> None:
        sent, log = [], []
        done = asyncio.Event()

        async def send(ev):
            sent.append(ev)
            if ev["type"] == "agent":
                done.set()

        controller = _controller({"send": send, "log": log}, FakeLlm())
        await controller.on_script("Hi, this call is transcribed.", turn_id=0)
        await asyncio.wait_for(done.wait(), timeout=0.5)
        assert [e["type"] for e in sent] == ["audio_end", "agent"]
        assert len([e for e in log if isinstance(e, tuple)]) == 1  # audio released

    asyncio.run(run())
