"""Reply orchestration, playback drain, speculation, and delivery tests."""
import asyncio
import time
from types import SimpleNamespace

import server.realtime.reply as reply_module
from server.engine.stub import StubEngine
from server.realtime.bargein import BargeInGuard
from server.realtime.endpoint import TurnComplete
from server.realtime.reply import (
    DRAIN_ACK_MARGIN_SEC,
    DRAIN_ACK_MAX_SEC,
    drain_wait_seconds,
    wait_for_playback_drain,
)
from server.realtime.supervisor import GenerationSupervisor

def test_speculation_identity_preserves_meaningful_punctuation() -> None:
    assert reply_module._spoken_eq("Five years", "five years.")
    assert not reply_module._spoken_eq("No, nights.", "No nights.")


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
        async def speak(self, *_args, **_kwargs):
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
        controller._audible_generation = None
        controller._metric_prefix = ""
        controller._call_id = "test-call"
        controller._state = SimpleNamespace(conversation=[])
        controller.guard = BargeInGuard()
        controller.interview = None
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
        async def speak(self, *_args, **_kwargs):
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
        controller._audible_generation = None
        controller._metric_prefix = ""
        controller._state = SimpleNamespace(conversation=[])
        controller.guard = BargeInGuard()
        controller.interview = None
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
    def __init__(self):  # parent initialization deliberately skipped
        self.transcripts = []
        self.extractions = []
        self.last_ttft_ms = None

    async def respond(self, transcript, **_kw):
        self.transcripts.append(transcript)
        yield f"Reply to: {transcript}"

    async def extract(self, transcript, **_kw):
        self.extractions.append(transcript)
        return []


class GatedSpeaker:
    def __init__(self, events):
        self.events = events

    async def speak(self, sentences, anchors, generation_id, is_current,
                    release=None, on_sentence=None):
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
    controller._audible_generation = None
    controller._browser_turn_anchors = {}
    controller._committed_turns = set()
    controller._extraction_tasks = set()
    controller._extraction_tail = None
    controller._closed = False
    controller._metric_prefix = ""
    controller._call_id = "test-call"
    controller._state = SimpleNamespace(conversation=[])
    controller.guard = BargeInGuard()
    controller.interview = None
    controller._engine = engine
    return controller


def test_voice_path_never_injects_server_authored_acknowledgements() -> None:
    async def run() -> None:
        async def send(_ev):
            pass

        controller = _controller({"send": send, "log": []}, FakeLlm())
        token = await controller._supervisor.start(
            1, lambda _owned: asyncio.sleep(10))
        lines = [line async for line in controller._voiced_sentences(
            token, "Delaware.")]
        await controller._supervisor.cancel_current()
        assert lines == ["Reply to: Delaware."]

    asyncio.run(run())


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
        assert controller._engine.extractions == ["Hello there."]

    asyncio.run(run())


def test_committed_extraction_survives_agent_reply_interruption() -> None:
    class SlowExtractionLlm(FakeLlm):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.applied = False

        async def extract(self, transcript, **_kw):
            self.started.set()
            await self.release.wait()
            self.applied = True
            return []

    async def run() -> None:
        async def send(_ev):
            pass

        engine = SlowExtractionLlm()
        controller = _controller({"send": send, "log": []}, engine)
        token = await controller._supervisor.start(
            8, lambda _owned: asyncio.sleep(10))
        controller._queue_extraction(token, "Night shifts work for me.")
        await engine.started.wait()

        # Barge-in invalidates agent playback, not the committed caller turn.
        await controller._supervisor.cancel_current()
        engine.release.set()
        await controller._extraction_tail
        assert engine.applied is True

    asyncio.run(run())


def test_committed_extractions_run_in_caller_turn_order() -> None:
    class OrderedExtractionLlm(FakeLlm):
        def __init__(self):
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def extract(self, transcript, **_kw):
            self.extractions.append(transcript)
            if transcript == "first answer":
                self.first_started.set()
                await self.release_first.wait()
            return []

    async def run() -> None:
        async def send(_ev):
            pass

        engine = OrderedExtractionLlm()
        controller = _controller({"send": send, "log": []}, engine)
        first = await controller._supervisor.start(
            10, lambda _owned: asyncio.sleep(10))
        controller._queue_extraction(first, "first answer")
        await engine.first_started.wait()

        await controller._supervisor.cancel_current()
        second = await controller._supervisor.start(
            11, lambda _owned: asyncio.sleep(10))
        controller._queue_extraction(second, "second answer")
        await asyncio.sleep(0)
        assert engine.extractions == ["first answer"]

        engine.release_first.set()
        await controller._extraction_tail
        assert engine.extractions == ["first answer", "second answer"]
        await controller._supervisor.cancel_current()

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


def test_speculative_failure_is_not_visible_before_commit() -> None:
    class ExplodingDraftSpeaker(GatedSpeaker):
        async def speak(self, *_args, **_kwargs):
            raise RuntimeError("draft provider failed")

    async def run() -> None:
        sent, log = [], []

        async def send(ev):
            sent.append(ev)

        controller = _controller({"send": send, "log": log}, FakeLlm())
        controller._speaker = ExplodingDraftSpeaker(log)
        await controller.speculate("Finished.", turn_id=3)
        await asyncio.sleep(0.05)
        assert sent == []
        assert controller._state.conversation == []

    asyncio.run(run())


def test_tool_failures_are_returned_to_reply_orchestration() -> None:
    class RejectingLlm(FakeLlm):
        def __init__(self):
            super().__init__()
            self.last_tool_results = []

        async def extract(self, transcript, **_kw):
            self.extractions.append(transcript)
            return [{
                "tool_call_id": "tc1", "name": "record_answer",
                "applied": False, "reason": "quote not supported",
            }]

    async def run() -> None:
        sent, log = [], []

        async def send(ev):
            sent.append(ev)

        controller = _controller({"send": send, "log": log}, RejectingLlm())
        token = await controller._supervisor.start(7, lambda _owned: asyncio.sleep(10))
        lines = [line async for line in controller._sentences_for(token, "unclear")]
        controller._queue_extraction(token, "unclear")
        await controller._extraction_tail
        await controller._supervisor.cancel_current()
        assert lines == ["Reply to: unclear"]
        assert sent[0]["type"] == "tool_failures"
        assert "arguments" not in sent[0]["failures"][0]

    asyncio.run(run())


def test_browser_playback_start_records_audible_latency(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        reply_module.registry, "record_turn",
        lambda call_id, **values: recorded.append((call_id, values)))

    async def run() -> None:
        async def send(_ev):
            pass

        controller = _controller({"send": send, "log": []}, FakeLlm())
        token = await controller._supervisor.start(2, lambda _owned: asyncio.sleep(10))
        controller._remember_browser_anchor(
            token.generation_id, asyncio.get_running_loop().time() - 0.05)
        await controller.on_playback_started(token.generation_id)
        await controller._supervisor.cancel_current()
        assert recorded[0][0] == "test-call"
        assert recorded[0][1]["browser_turn_latency_ms"] >= 50

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


def test_controller_selects_tts_provider_from_settings(monkeypatch) -> None:
    from server.config import Settings
    from server.realtime.tts_aura import AuraTts

    async def no_connect(self) -> None:
        return None

    monkeypatch.setattr(AuraTts, "ensure_connected", no_connect)

    async def run() -> None:
        async def send(_ev):
            pass

        settings = Settings(_env_file=None, deepgram_api_key="dg",
                            tts_provider="aura")
        controller = reply_module.ReplyController(
            send, send, settings, SimpleNamespace(conversation=[]))
        assert isinstance(controller._tts, AuraTts)
        assert controller._speaker is not None
        controller._prewarm.cancel()

        # elevenlabs stays the default path when its key is present
        settings = Settings(_env_file=None, elevenlabs_api_key="el")
        controller = reply_module.ReplyController(
            send, send, settings, SimpleNamespace(conversation=[]))
        assert isinstance(controller._tts, reply_module.MultiContextTts)
        controller._prewarm.cancel()

    asyncio.run(run())


def test_agent_sentences_reach_the_client_while_she_is_still_speaking(monkeypatch) -> None:
    # Live report: the agent's transcript card only appeared once she finished
    # the whole reply. Each sentence must land as its audio starts, with the
    # final "agent" event closing the turn.
    _spec_env(monkeypatch)

    class AnnouncingSpeaker:
        async def speak(self, sentences, anchors, generation_id, is_current,
                        release=None, on_sentence=None):
            async for sentence in sentences:
                await on_sentence(sentence)
            return {"_playback_remaining_sec": 0.0}

        def text(self, _generation_id):
            return "One. Two."

    async def run() -> None:
        sent, log = [], []
        done = asyncio.Event()

        async def send(ev):
            sent.append(ev)
            if ev["type"] == "agent":
                done.set()

        class TwoSentenceLlm(FakeLlm):
            async def respond(self, transcript, **_kw):
                yield "One."
                yield "Two."

        controller = _controller({"send": send, "log": log}, TwoSentenceLlm())
        controller._speaker = AnnouncingSpeaker()
        now = time.monotonic()
        turn = TurnComplete("Hi.", 0.05, now, now, "fast")
        await controller.on_turn(turn, 1)
        await asyncio.wait_for(done.wait(), timeout=0.5)

        partials = [e for e in sent if e["type"] == "agent_partial"]
        assert [e["text"] for e in partials] == ["One.", "One. Two."]
        assert all(e["generation_id"] == partials[0]["generation_id"]
                   for e in partials)
        # ordering: every partial precedes the committed agent card
        assert sent.index(partials[-1]) < sent.index(
            next(e for e in sent if e["type"] == "agent"))

    asyncio.run(run())


def test_caller_noise_before_any_audio_does_not_kill_the_reply(monkeypatch) -> None:
    # Found live the moment the mic started working again: the guard was armed
    # when the GENERATION started, but first audio is ~2 s later (LLM + TTS
    # connect). Any 250 ms of room noise in that window cancelled a reply the
    # caller had never heard — the call just sat there silent.
    _spec_env(monkeypatch)

    class SlowSpeaker:
        def __init__(self):
            self.cancelled = False

        async def speak(self, sentences, anchors, generation_id, is_current,
                        release=None, on_sentence=None):
            try:
                await asyncio.sleep(5)  # still connecting: nothing sent yet
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return {"_playback_remaining_sec": 0.0}

        def text(self, _generation_id):
            return "never got here"

    async def run() -> None:
        async def send(_ev):
            pass

        controller = _controller({"send": send, "log": []}, FakeLlm())
        speaker = SlowSpeaker()
        controller._speaker = speaker

        await controller.on_script("Hi, this is Sarah.", turn_id=0)
        await asyncio.sleep(0.05)

        # The caller makes a sound and keeps making it past the onset window.
        controller.guard.on_vad_start(0.0)
        fired = controller.guard.tick(0.0 + controller.guard.onset_sec)
        assert not fired, "nothing audible has been sent; this is not a barge-in"
        assert not speaker.cancelled

        await controller._supervisor.cancel_current()

    asyncio.run(run())


def test_barge_in_still_fires_once_audio_is_actually_going_out(monkeypatch) -> None:
    _spec_env(monkeypatch)

    async def run() -> None:
        async def send(_ev):
            pass

        sent_frames = []

        async def send_bytes(payload):
            sent_frames.append(payload)

        controller = _controller({"send": send, "log": []}, FakeLlm())
        controller._send_bytes = send_bytes

        token = await controller._supervisor.start(1, lambda _t: asyncio.sleep(5))
        controller._active_voice_generation = token.generation_id
        await controller._send_audio(token.generation_id, bytes(640))

        assert controller.guard.agent_speaking, "armed by real audio going out"
        controller.guard.on_vad_start(0.0)
        assert controller.guard.tick(controller.guard.onset_sec) is True

        await controller._supervisor.cancel_current()

    asyncio.run(run())


def test_script_copy_is_spoken_and_announced_sentence_by_sentence(monkeypatch) -> None:
    # The disclosure was handed to TTS as ONE paragraph, so its whole text hit
    # the transcript the instant her first frame went out (measured live: text
    # at 2.6 s, she stopped talking at 16.7 s) and one utterance had to carry
    # ~14 s of audio. Chunk it like engine output instead.
    _spec_env(monkeypatch)

    class RecordingSpeaker:
        def __init__(self):
            self.spoken = []

        async def speak(self, sentences, anchors, generation_id, is_current,
                        release=None, on_sentence=None):
            async for sentence in sentences:
                self.spoken.append(sentence)
                if on_sentence is not None:
                    await on_sentence(sentence)
            return {"_playback_remaining_sec": 0.0}

        def text(self, _generation_id):
            return " ".join(self.spoken)

    async def run() -> None:
        sent = []
        done = asyncio.Event()

        async def send(ev):
            sent.append(ev)
            if ev["type"] == "agent":
                done.set()

        controller = _controller({"send": send, "log": []}, FakeLlm())
        speaker = RecordingSpeaker()
        controller._speaker = speaker
        text = ("Hi, this is Sarah. I will transcribe what you say. "
                "No audio recording is stored. Do you consent to continue now?")
        await controller.on_script(text, turn_id=0)
        await asyncio.wait_for(done.wait(), timeout=0.5)

        assert speaker.spoken == [
            "Hi, this is Sarah.",
            "I will transcribe what you say.",
            "No audio recording is stored.",
            "Do you consent to continue now?",
        ]
        partials = [e["text"] for e in sent if e["type"] == "agent_partial"]
        assert len(partials) == 4                  # transcript grows with her
        assert partials[0] == "Hi, this is Sarah."
        assert partials[-1] == text                # and ends up complete
        committed = next(e for e in sent if e["type"] == "agent")
        assert committed["text"] == text           # nothing lost in the split

    asyncio.run(run())


def test_controller_warms_the_engine_connection_during_the_greeting(monkeypatch) -> None:
    from server.config import Settings

    warmed = asyncio.Event()

    async def fake_warm(self) -> None:
        warmed.set()

    monkeypatch.setattr(reply_module.LlmEngine, "warm_connection", fake_warm)

    async def run() -> None:
        async def send(_ev):
            pass

        settings = Settings(_env_file=None, openai_api_key="sk-test")
        controller = reply_module.ReplyController(
            send, send, settings, SimpleNamespace(conversation=[]))
        await asyncio.wait_for(warmed.wait(), timeout=0.5)
        await controller.close()          # must cancel cleanly, not warn

    asyncio.run(run())
