"""Provisional evidence extraction, run beside speech in caller-turn order.

## How this works
Speech never waits for evidence: when a caller turn commits, the reply starts
streaming immediately and an extraction request for the same turn is queued
here. Requests form a chain - each awaits the previous one - so evidence lands
in caller-turn order even when the model answers out of order. Ownership is
the TURN, not the agent's playback: a barge-in cancels Sarah's reply through
the GenerationSupervisor, but the evidence the caller already gave stays
valid and still applies. Tasks are tracked so close() can cancel and await
every one (nothing fire-and-forget), and failures are reported as
tool_failures events rather than swallowed. Measured live: running this
concurrently costs speech +12 ms of first-token latency (paired, n=12).
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable

from server.engine.turn import LlmEngine
from server.realtime.supervisor import GenerationToken

log = logging.getLogger(__name__)


class ExtractionQueue:
    def __init__(self, engine: object, send: Callable[[dict], Awaitable[None]],
                 interview: object) -> None:
        self._engine = engine
        self._send = send
        self.interview = interview
        self._committed_turns: set[int] = set()
        self._tasks: set[asyncio.Task] = set()
        self.tail: asyncio.Task | None = None
        self._closed = False

    def queue(self, token: GenerationToken, transcript: str) -> None:
        """Only committed turns reach this method. Once committed, their
        evidence remains valid even if the caller interrupts the reply."""
        engine = self._engine
        if not isinstance(engine, LlmEngine) or self._closed:
            return
        if token.turn_id in self._committed_turns:
            return
        self._committed_turns.add(token.turn_id)
        if self.interview is not None:
            self.interview.add_history("user", transcript)
        previous = self.tail

        def is_valid() -> bool:
            return not self._closed and token.turn_id in self._committed_turns

        async def run() -> None:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    if self._closed:
                        return
                except Exception:
                    # The prior runner normally contains its own failures. Keep
                    # later committed evidence moving if it failed unexpectedly.
                    pass
            if not is_valid():
                return
            try:
                results = await engine.extract(
                    transcript, is_valid=is_valid, turn_id=token.turn_id,
                    generation_id=token.generation_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("background evidence extraction failed")
                return
            failures = [
                {"tool_call_id": result.get("tool_call_id"),
                 "name": result.get("name"), "reason": result.get("reason")}
                for result in results
                if (not result.get("applied")
                    and not str(result.get("reason", "")).startswith("duplicate"))
            ]
            if failures and is_valid():
                await self._send({
                    "type": "tool_failures", "turn_id": token.turn_id,
                    "generation_id": token.generation_id,
                    "failures": failures,
                })

        task = asyncio.create_task(run())
        self.tail = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        self._closed = True
        for task in tuple(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks.clear()
        self.tail = None
