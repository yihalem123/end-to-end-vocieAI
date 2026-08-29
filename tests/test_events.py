"""EventBuffer: replaceable partials are evictable; finals/control never drop."""
import asyncio

from server.realtime.asr import AsrFinal, AsrPartial, AsrUtteranceEnd
from server.realtime.events import EventBuffer
from server.realtime.flux import FluxEndOfTurn, FluxUpdate


def drain(buf: EventBuffer) -> list:
    async def run() -> list:
        out = []
        while buf.pending:
            out.append(await buf.get())
        return out
    return asyncio.run(run())


def test_fifo_order_preserved() -> None:
    buf = EventBuffer()
    events = [AsrPartial("a"), AsrFinal("a.", True), AsrUtteranceEnd()]
    for e in events:
        buf.put_nowait(e)
    assert drain(buf) == events


def test_overflow_evicts_oldest_partial_never_finals() -> None:
    buf = EventBuffer(replaceable_limit=2)
    buf.put_nowait(AsrPartial("one"))
    buf.put_nowait(AsrFinal("kept.", True))
    buf.put_nowait(AsrPartial("two"))
    buf.put_nowait(AsrPartial("three"))     # over limit: "one" is evicted
    out = drain(buf)
    assert AsrFinal("kept.", True) in out
    assert AsrPartial("one") not in out
    assert out[-1] == AsrPartial("three")   # newest partial survives
    assert buf.replaced == 1


def test_flux_updates_are_replaceable_too() -> None:
    buf = EventBuffer(replaceable_limit=1)
    buf.put_nowait(FluxUpdate("hel"))
    buf.put_nowait(FluxUpdate("hello"))     # replaces the stale partial
    buf.put_nowait(FluxEndOfTurn("hello."))
    out = drain(buf)
    assert out == [FluxUpdate("hello"), FluxEndOfTurn("hello.")]


def test_criticals_always_admitted_beyond_replaceable_limit() -> None:
    buf = EventBuffer(replaceable_limit=1)
    finals = [AsrFinal(f"s{i}.", False) for i in range(10)]
    for f in finals:
        buf.put_nowait(f)
    assert drain(buf) == finals             # nothing critical was lost
    assert buf.replaced == 0


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
    buf = EventBuffer(replaceable_limit=1)
    buf.put_nowait({"kind": "client"})      # arbitrary pipeline event
    buf.put_nowait({"kind": "client2"})
    assert drain(buf) == [{"kind": "client"}, {"kind": "client2"}]
