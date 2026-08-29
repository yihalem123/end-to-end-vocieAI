"""Generation supervisor serialization and stale-event ownership."""
import asyncio

from server.realtime.supervisor import GenerationSupervisor


def test_replacement_cancels_and_awaits_before_new_runner() -> None:
    async def run() -> list[str]:
        supervisor = GenerationSupervisor()
        events: list[str] = []
        blocker = asyncio.Event()

        async def first(_token) -> None:
            events.append("first-start")
            try:
                await blocker.wait()
            finally:
                await asyncio.sleep(0)
                events.append("first-clean")

        async def second(_token) -> None:
            events.append("second-start")

        old = await supervisor.start(1, first)
        await asyncio.sleep(0)
        new = await supervisor.start(2, second)
        await asyncio.sleep(0)
        assert not supervisor.is_current(old)
        assert new.generation_id == old.generation_id + 1
        return events

    assert asyncio.run(run()) == ["first-start", "first-clean", "second-start"]


def test_only_explicitly_interrupted_generation_accepts_clear() -> None:
    async def run() -> None:
        supervisor = GenerationSupervisor()

        async def forever(_token) -> None:
            await asyncio.Event().wait()

        old = await supervisor.start(1, forever)
        await asyncio.sleep(0)
        token, task = supervisor.begin_interrupt()
        assert token == old
        assert not supervisor.is_current(old)
        await supervisor.wait_cancelled(task)
        assert supervisor.accepts_clear(old.generation_id)
        assert not supervisor.accepts_clear(old.generation_id + 1)

        await supervisor.start(2, forever)
        assert not supervisor.accepts_clear(old.generation_id)
        await supervisor.close()

    asyncio.run(run())


def test_failed_previous_worker_does_not_block_replacement() -> None:
    async def run() -> None:
        supervisor = GenerationSupervisor()
        ran = asyncio.Event()

        async def broken(_token) -> None:
            raise RuntimeError("old generation failed")

        async def replacement(_token) -> None:
            ran.set()

        await supervisor.start(1, broken)
        await asyncio.sleep(0)
        await supervisor.start(2, replacement)
        await asyncio.sleep(0)
        assert ran.is_set()

    asyncio.run(run())
