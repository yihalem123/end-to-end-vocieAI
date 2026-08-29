"""EventBuffer: reliable priority plus one latest replaceable partial."""
import asyncio
import pytest

from server.realtime.asr import AsrFinal, AsrPartial, AsrUtteranceEnd
from server.realtime.events import CriticalEventOverflow, EventBuffer
from server.realtime.flux import FluxEndOfTurn, FluxUpdate


def drain(buf: EventBuffer) -> list:
    async def run() -> list:
        out = []
        while buf.pending:
            out.append(await buf.get())
        return out
    return asyncio.run(run())


def test_reliable_events_jump_ahead_of_partial() -> None:
    buf = EventBuffer()
    events = [AsrPartial("a"), AsrFinal("a.", True), AsrUtteranceEnd()]
    for e in events:
        buf.put_nowait(e)
    assert drain(buf) == [events[1], events[2], events[0]]


def test_every_new_partial_replaces_the_previous_one() -> None:
    buf = EventBuffer()
    buf.put_nowait(AsrPartial("one"))
    buf.put_nowait(AsrFinal("kept.", True))
    buf.put_nowait(AsrPartial("two"))
    buf.put_nowait(AsrPartial("three"))     # over limit: "one" is evicted
    out = drain(buf)
    assert out == [AsrFinal("kept.", True), AsrPartial("three")]
    assert buf.replaced == 2


def test_flux_updates_are_replaceable_too() -> None:
    buf = EventBuffer()
    buf.put_nowait(FluxUpdate("hel"))
    buf.put_nowait(FluxUpdate("hello"))     # replaces the stale partial
    buf.put_nowait(FluxEndOfTurn("hello."))
    out = drain(buf)
    assert out == [FluxEndOfTurn("hello."), FluxUpdate("hello")]


def test_reliable_lane_is_bounded_and_fails_loudly() -> None:
    buf = EventBuffer(reliable_limit=2)
    buf.put_nowait(AsrFinal("s1.", False))
    buf.put_nowait(AsrFinal("s2.", False))
    with pytest.raises(CriticalEventOverflow):
        buf.put_nowait(AsrFinal("s3.", False))


def test_get_waits_for_next_event() -> None:
    async def run() -> object:
        buf = EventBuffer()

        async def put_later() -> None:
            await asyncio.sleep(0.01)
            buf.put_nowait(AsrUtteranceEnd())

        asyncio.get_running_loop().create_task(put_later())
        return await asyncio.wait_for(buf.get(), timeout=1.0)

    assert asyncio.run(run()) == AsrUtteranceEnd()


def test_non_asr_events_are_critical() -> None:
    buf = EventBuffer(reliable_limit=2)
    buf.put_nowait({"kind": "client"})      # arbitrary pipeline event
    buf.put_nowait({"kind": "client2"})
    assert drain(buf) == [{"kind": "client"}, {"kind": "client2"}]
