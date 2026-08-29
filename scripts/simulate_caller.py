"""Simulated caller: a scripted persona speaks to the running server. Phase 5b.

## How this works
The rambling persona is an endpointer stress test: each answer is split into
parts with a deliberate mid-answer PAUSE (1.5 s of silence) between them, and
the leading part ends in a trailing word ("um,", "at,") — exactly the shape
that used to get callers cut off. Audio is synthesized offline with Windows
SAPI (16 kHz mono PCM16, the wire format — no vendor cost), then streamed as
real-time-paced 20 ms frames. The script waits for each agent reply before
answering. It decodes generation-prefixed agent audio and acknowledges clear or
fully received playback just like the browser, then hangs up, fetches the
post-call report, and ASSERTS:
  1. no answer was split by a premature commit (turns == answers),
  2. the trailing/slow patience tiers actually fired,
  3. extraction got the facts right despite the rambling.
Exit code 0 = all assertions pass. Run: python scripts/simulate_caller.py
(server must be running; needs Deepgram + OpenAI keys, ElevenLabs optional).
Run `python scripts/simulate_caller.py --protocol-self-test` for the offline
wire/ack verification used in CI and local review.

Usage note: this is a script, not pytest — it exercises live vendors end to
end, which does not belong in the unit suite.
"""
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.realtime.protocol import decode_audio_frame, encode_audio_frame  # noqa: E402

SERVER = "127.0.0.1:8080"
MODE = "flux" if "--flux" in sys.argv else "custom"  # same rambler, both stacks
FRAME_BYTES = 640
FRAME_SEC = 0.02
PAUSE = object()  # mid-answer silence: the endpointer patience test.
# 1.0 s scripted ≈ 1.5-2.0 s effective gap (VAD stop hysteresis + synth edges);
# that must stay UNDER the 2.5 s trailing patience or splitting is by design.
PAUSE_SEC = 1.0


@dataclass
class Answer:
    parts: list  # str segments and PAUSE markers


@dataclass
class PlaybackTracker:
    """Protocol-only playback model used by the headless live harness."""

    played_samples: dict[int, int] = field(default_factory=dict)

    def on_audio(self, wire: bytes) -> tuple[int, bytes]:
        generation_id, pcm = decode_audio_frame(wire)
        self.played_samples[generation_id] = (
            self.played_samples.get(generation_id, 0) + len(pcm) // 2)
        return generation_id, pcm

    def acknowledgement(self, event: dict) -> dict | None:
        generation_id = int(event.get("generation_id", 0))
        if generation_id <= 0:
            return None
        if event.get("type") == "clear":
            return {
                "type": "cleared",
                "generation_id": generation_id,
                "played_samples": self.played_samples.get(generation_id, 0),
            }
        if event.get("type") == "audio_end":
            return {"type": "playback_drained", "generation_id": generation_id}
        return None


# Part texts deliberately avoid internal commas: SAPI renders commas as
# ~700 ms dramatic pauses, and a punctuated fragment plus 700 ms of silence is
# a LEGITIMATE fast commit ("Yeah." then silence is a finished answer). The
# scripted PAUSE between parts is the one thing under test.
RAMBLER = [
    Answer(["Hello?"]),
    # Ends mid-clause ("if"): an UNAMBIGUOUS continuation. "Yeah, sure." plus a
    # pause is legitimately committable (sentence-final prosody, complete
    # affirmative) — the contract only promises to hold clear mid-thought pauses.
    Answer(["Yeah sure I mean if", PAUSE, "if now works then yes go ahead."]),
    Answer(["So my license is um", PAUSE, "it's in Texas and it is active."]),
    Answer(["I've been in the I C U for about uh", PAUSE,
            "six years maybe six and a half."]),
    Answer(["I've got my B L S and", PAUSE, "A C L S and I'm working on the C C R N."]),
    Answer(["Honestly what I prefer is", PAUSE, "nights are fine for me."]),
    Answer(["I'd have to give notice so like", PAUSE, "three weeks from now."]),
    Answer(["Somewhere around um", PAUSE, "fifty five dollars an hour."]),
]


def synth(text: str, path: Path) -> None:
    safe = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $fmt); $s.Speak('{safe}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   check=True, capture_output=True)


def load_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())
    return trim_silence(pcm)


def trim_silence(pcm: bytes, threshold: int = 300) -> bytes:
    """Cut SAPI's leading/trailing silence padding so scripted pause lengths
    mean what they say (untrimmed edges stretched a 1.5 s pause past the
    2.5 s patience tier and split turns by design, not by bug)."""
    import array
    samples = array.array("h", pcm)
    loud = [i for i, s in enumerate(samples) if abs(s) > threshold]
    if not loud:
        return pcm
    start, end = max(0, loud[0] - 160), min(len(samples), loud[-1] + 160)
    return samples[start:end].tobytes()


async def stream_answer(ws, answer: Answer, cache: dict) -> None:
    for part in answer.parts:
        pcm = bytes(int(PAUSE_SEC * 32000)) if part is PAUSE else cache[part]
        for off in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
            await ws.send(pcm[off:off + FRAME_BYTES])
            await asyncio.sleep(FRAME_SEC)
    # trailing silence so the endpointer can commit the turn
    for _ in range(150):
        await ws.send(bytes(FRAME_BYTES))
        await asyncio.sleep(FRAME_SEC)


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="simcaller_"))
    cache: dict = {}
    for answer in RAMBLER:
        for part in answer.parts:
            if part is not PAUSE and part not in cache:
                wav = tmp / f"{abs(hash(part))}.wav"
                synth(part, wav)
                cache[part] = load_pcm(wav)
    print(f"synthesized {len(cache)} segments")

    turns: list[dict] = []
    vad_stops = 0
    agent_events = asyncio.Queue()
    playback = PlaybackTracker()
    print(f"mode: {MODE}")
    async with websockets.connect(f"ws://{SERVER}/ws/call?mode={MODE}") as ws:
        async def reader() -> None:
            async for msg in ws:
                if isinstance(msg, bytes):
                    playback.on_audio(msg)  # strips generation header and counts PCM
                    continue
                ev = json.loads(msg)
                ack = playback.acknowledgement(ev)
                if ack is not None:
                    await ws.send(json.dumps(ack))
                if ev["type"] == "vad" and ev["state"] == "silence":
                    nonlocal vad_stops
                    vad_stops += 1
                if ev["type"] == "turn":
                    turns.append(ev)
                    print(f"  turn ({ev['endpoint_delay_ms']} ms, {ev['reason']}): "
                          f"\"{ev['transcript'][:70]}\"")
                elif ev["type"] == "agent":
                    print(f"  agent: \"{ev['text'][:70]}\"")
                    await agent_events.put(ev)

        reader_task = asyncio.create_task(reader())
        for i, answer in enumerate(RAMBLER):
            await stream_answer(ws, answer, cache)
            try:
                await asyncio.wait_for(agent_events.get(), timeout=45)
            except TimeoutError:
                print(f"FAIL: no agent reply after answer {i}")
                return 1
        reader_task.cancel()
        with suppress(asyncio.CancelledError):
            await reader_task

    await asyncio.sleep(12)  # post-call extraction
    async with httpx.AsyncClient() as client:
        calls = (await client.get(f"http://{SERVER}/calls")).json()
        report = (await client.get(
            f"http://{SERVER}/report/{calls[0]['call_id']}")).json()

    failures: list[str] = []
    if len(turns) != len(RAMBLER):
        failures.append(f"expected {len(RAMBLER)} turns, endpointer made {len(turns)} "
                        "(a mid-answer pause split a turn)")
    # Patience SUCCESS is invisible in commit reasons (a trailing-reason commit
    # means patience ran out). Proof the pauses tested anything: silences that
    # did NOT become commits — VAD stops must exceed turns by ~the pause count.
    held = vad_stops - len(turns)
    if held < 4:
        failures.append(f"only {held} mid-turn silences held — pauses tested nothing")
    fields = report.get("fields", {})
    if not (5.5 <= (fields.get("icu_years", {}).get("value") or 0) <= 7):
        failures.append(f"icu_years wrong: {fields.get('icu_years')}")
    certs = " ".join(fields.get("certifications", {}).get("value") or []).upper()
    if "BLS" not in certs or "ACLS" not in certs:
        failures.append(f"certifications wrong: {fields.get('certifications')}")
    if report.get("knocked_out"):
        failures.append(f"false knockout: {report['knocked_out']}")

    print(f"\nreport {report['call_id']}: score={report.get('score')} "
          f"needs_review={report.get('needs_review')}")
    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"PASS: {len(turns)} turns, no premature commits, extraction correct")
    return 0


def protocol_self_test() -> int:
    tracker = PlaybackTracker()
    generation_id, pcm = tracker.on_audio(encode_audio_frame(7, bytes(FRAME_BYTES)))
    assert generation_id == 7 and len(pcm) == FRAME_BYTES
    assert tracker.acknowledgement(
        {"type": "audio_end", "generation_id": 7}
    ) == {"type": "playback_drained", "generation_id": 7}
    assert tracker.acknowledgement({"type": "clear", "generation_id": 7}) == {
        "type": "cleared", "generation_id": 7, "played_samples": 320}
    print("PASS: generation audio decode and playback acknowledgements")
    return 0


if __name__ == "__main__":
    if "--protocol-self-test" in sys.argv:
        sys.exit(protocol_self_test())
    sys.exit(asyncio.run(main()))
