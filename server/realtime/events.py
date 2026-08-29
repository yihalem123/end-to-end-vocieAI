"""Priority event buffer: reliable events first, one replaceable partial.

## How this works
ASR partials are UI hints, not durable facts, so the buffer keeps only the most
recent partial. Finals, VAD transitions and playback acknowledgements use a
bounded reliable lane and are always consumed before the partial slot. A full
reliable lane fails loudly with CriticalEventOverflow rather than silently
losing a turn boundary or growing memory without limit. Producers never wait,
so provider socket readers cannot be back-pressured by browser rendering.
"""
import asyncio

from server.realtime.asr import AsrPartial
from server.realtime.flux import FluxUpdate

REPLACEABLE_TYPES = (AsrPartial, FluxUpdate)


class CriticalEventOverflow(RuntimeError):
    """The bounded reliable lane cannot safely accept another control event."""


class EventBuffer:
    def __init__(self, reliable_limit: int = 256) -> None:
        if reliable_limit < 1:
            raise ValueError("reliable_limit must be positive")
        self._reliable: asyncio.Queue = asyncio.Queue(maxsize=reliable_limit)
        self._partial = None
        self._ready = asyncio.Event()
        self.replaced = 0

    @property
    def pending(self) -> int:
        return self._reliable.qsize() + int(self._partial is not None)

    def put_nowait(self, event) -> None:
        if isinstance(event, REPLACEABLE_TYPES):
            if self._partial is not None:
                self.replaced += 1
            self._partial = event
            self._ready.set()
            return
        try:
            self._reliable.put_nowait(event)
        except asyncio.QueueFull as exc:
            raise CriticalEventOverflow(
                "reliable pipeline event lane exceeded bounded capacity"
            ) from exc
        self._ready.set()

    async def get(self):
        while True:
            try:
                return self._reliable.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self._partial is not None:
                event, self._partial = self._partial, None
                return event
            self._ready.clear()
            # No await occurs between the checks and clear, so a producer cannot
            # interleave here on the event loop and lose the wake-up.
            await self._ready.wait()
