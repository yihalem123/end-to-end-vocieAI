"""Barge-in onset guard: should the agent shut up?

## How this works
Same pattern as endpoint.py: a synchronous state machine fed explicit timestamps,
driven from call.py's tick loop, unit-tested with fake clocks. The question it
answers: "has the caller been speaking for >= onset_sec WHILE agent audio is
playing?" The onset guard exists because VAD start fires in ~64 ms on any voiced
sound — a cough or "mm" would kill the agent mid-sentence. Requiring sustained
speech (250 ms) trades a slightly later cut for far fewer false cuts; combined
with the client-side buffer flush the caller still perceives < 300 ms.

States: guard arms on vad_start while agent audio plays; disarms on vad_stop
(blip), on agent audio ending naturally (that's turn-taking, not interruption),
or after firing once. Re-arms fresh for the agent's next reply. tick() returns
True exactly once per genuine interruption; call.py then cancels TTS, tells the
client to flush, and truncates the transcript.

"While agent audio plays" means AUDIBLE audio: on_agent_audio_start fires on
the first frame actually sent (reply.py's _send_audio), not when the reply
generation starts. Those differ by ~2 s of LLM + TTS work, and arming early
let caller noise cancel replies that were never heard — a silent, dead-looking
call. Speech before the first frame is not an interruption; it commits a turn
and replaces the pending reply through the normal path.
"""


class BargeInGuard:
    def __init__(self, onset_sec: float = 0.25) -> None:
        self.onset_sec = onset_sec
        self._agent_speaking = False
        self._onset_t: float | None = None  # armed at this time, None = disarmed

    @property
    def agent_speaking(self) -> bool:
        return self._agent_speaking

    def on_agent_audio_start(self) -> None:
        self._agent_speaking = True
        self._onset_t = None

    def on_agent_audio_end(self) -> None:
        self._agent_speaking = False
        self._onset_t = None

    def on_vad_start(self, t: float) -> None:
        if self._agent_speaking:
            self._onset_t = t

    def on_vad_stop(self, t: float) -> None:
        self._onset_t = None

    def tick(self, t: float) -> bool:
        if self._onset_t is None or not self._agent_speaking:
            return False
        if t - self._onset_t >= self.onset_sec:
            # Fire once: the cut is happening; disarm until the next reply.
            self._onset_t = None
            self._agent_speaking = False
            return True
        return False
