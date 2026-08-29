"""Pipeline event buffer: partials are replaceable, everything else is not.

## How this works
The old event queue treated all events equally and dropped the NEWEST on
overflow — which could silently discard an ASR final or a client control ack,
the events the endpointer and truncation logic cannot function without. This
buffer encodes the real policy: interim transcripts (AsrPartial, FluxUpdate)
are REPLACEABLE — each supersedes the last, so under pressure the oldest
partial is evicted to make room. Finals, turn events, VAD transitions and
client acks are CRITICAL: always admitted, even past the soft bound (they are
rare and small; a hard cap with loud accounting beats silent loss). put_nowait
never raises and never blocks — the producers are socket readers that must not
stall. Single consumer (the call event loop) via get().
"""
import asyncio
from collections import deque

from server.realtime.asr import AsrPartial
from server.realtime.flux import FluxUpdate

REPLACEABLE_TYPES = (AsrPartial, FluxUpdate)


class EventBuffer:
    def __init__(self, replaceable_limit: int = 64) -> None:
        self._dq: deque = deque()
        self._limit = replaceable_limit
        self._replaceable_count = 0
        self._ready = asyncio.Event()
        self.replaced = 0  # stale partials evicted under pressure

    @property
    def pending(self) -> int:
        return len(self._dq)

    def put_nowait(self, event) -> None:
        if isinstance(event, REPLACEABLE_TYPES):
            if self._replaceable_count >= self._limit:
                self._evict_oldest_replaceable()
            self._replaceable_count += 1
        self._dq.append(event)
        self._ready.set()

    def _evict_oldest_replaceable(self) -> None:
        for i, ev in enumerate(self._dq):
            if isinstance(ev, REPLACEABLE_TYPES):
                del self._dq[i]
                self._replaceable_count -= 1
                self.replaced += 1
                return

    async def get(self):
        while not self._dq:
            self._ready.clear()
            await self._ready.wait()
        event = self._dq.popleft()
        if isinstance(event, REPLACEABLE_TYPES):
            self._replaceable_count -= 1
        return event
