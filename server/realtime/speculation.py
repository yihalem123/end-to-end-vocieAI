"""One speculative draft: start it early, promote it on match, cancel otherwise.

## How this works
A draft is a full reply generation (LLM + TTS) started on a GUESS of the
caller's finished utterance, before the endpointer commits. Safety comes from
three gates, not from the guess being right: audio is release-gated (nothing
reaches the caller until the real commit), the supervisor cancels the
generation if the caller resumes or the transcript changes, and history and
evidence apply only after promotion. At commit, take_if_matches() compares
the committed transcript with the guess - casing, spacing and final
punctuation may differ; internal punctuation may not ("No, nights" is not
"No nights") - and hands the draft back to the controller to release, or
returns None so a normal reply starts. Measured: custom mode gets no head
start (finals land at the commit moment; 0/10 pre-final drafts promoted), so
this pays only in Flux mode, where EagerEndOfTurn signals ahead of the final.
"""
import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

from server.realtime.supervisor import GenerationSupervisor, GenerationToken

log = logging.getLogger(__name__)


def spoken_eq(a: str, b: str) -> bool:
    """Allow only casing/space and final punctuation changes on promotion."""
    def norm(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".!?")

    return norm(a) == norm(b)


Deliver = Callable[[GenerationToken, str, dict, asyncio.Event], Awaitable[None]]


class SpeculationSlot:
    def __init__(self, supervisor: GenerationSupervisor) -> None:
        self._supervisor = supervisor
        self.current: dict | None = None  # {transcript, token, release, anchors}

    async def start(self, transcript: str, turn_id: int, deliver: Deliver) -> None:
        if self.current is not None:
            if spoken_eq(self.current["transcript"], transcript):
                return  # already speculating on effectively this turn
            await self.cancel()  # transcript grew: guess is stale
        if self._supervisor.current is not None:
            return  # a real reply is active; never preempt it on a guess
        release = asyncio.Event()
        anchors = {"commit_t": 0.0, "vad_stop_t": 0.0}
        spec = {"transcript": transcript, "release": release, "anchors": anchors}
        log.info("speculation START %r", transcript[:60])
        spec["token"] = await self._supervisor.start(
            turn_id, lambda owned: deliver(owned, transcript, anchors, release))
        self.current = spec

    async def cancel(self) -> None:
        spec, self.current = self.current, None
        if spec is not None and self._supervisor.is_current(spec["token"]):
            log.info("speculation CANCELLED %r", spec["transcript"][:60])
            await self._supervisor.cancel_current()

    def take_if_matches(self, turn_id: int, transcript: str) -> dict | None:
        """The committed turn is the guessed turn: hand the draft over."""
        spec = self.current
        if (spec is None
                or not self._supervisor.is_current(spec["token"])
                or spec["token"].turn_id != turn_id
                or not spoken_eq(spec["transcript"], transcript)):
            return None
        self.current = None
        log.info("speculation PROMOTED gen=%d", spec["token"].generation_id)
        return spec
