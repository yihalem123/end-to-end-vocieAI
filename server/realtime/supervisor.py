"""Serialized per-call turn/generation ownership.

## How this works
Each committed caller turn gets a monotonic turn id and each reply attempt gets a
monotonic generation id. Exactly one generation may be current. Replacement or
barge-in invalidates the token *before* cancelling its task, so stale cleanup can
never pass an `is_current` check. Cancellation is always awaited before a new
generation starts. A cleared acknowledgement is accepted only for the one
generation explicitly pending a browser flush; starting a new generation makes
any delayed acknowledgement stale.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationToken:
    turn_id: int
    generation_id: int


Runner = Callable[[GenerationToken], Awaitable[None]]


class GenerationSupervisor:
    def __init__(self) -> None:
        self._next_generation_id = 0
        self._current: GenerationToken | None = None
        self._task: asyncio.Task | None = None
        self._pending_clear: GenerationToken | None = None

    @property
    def current(self) -> GenerationToken | None:
        return self._current

    def is_current(self, token: GenerationToken) -> bool:
        return self._current == token

    async def start(self, turn_id: int, runner: Runner) -> GenerationToken:
        await self.cancel_current()
        self._pending_clear = None
        self._next_generation_id += 1
        token = GenerationToken(turn_id=turn_id,
                                generation_id=self._next_generation_id)
        self._current = token

        async def owned() -> None:
            try:
                await runner(token)
            finally:
                self.finish(token)

        self._task = asyncio.create_task(owned())
        return token

    def begin_interrupt(self) -> tuple[GenerationToken | None, asyncio.Task | None]:
        """Invalidate now; caller sends clear, then awaits the returned task."""
        token, task = self._current, self._task
        if token is None:
            return None, None
        self._current = None
        self._task = None
        self._pending_clear = token
        if task is not None and not task.done():
            task.cancel()
        return token, task

    async def wait_cancelled(self, task: asyncio.Task | None) -> None:
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # worker already reports user-facing failure; replacement continues
            log.exception("generation worker failed during cleanup")

    async def cancel_current(self) -> None:
        token, task = self._current, self._task
        self._current = None
        self._task = None
        if token is not None and task is not None and not task.done():
            task.cancel()
        await self.wait_cancelled(task)

    def accepts_clear(self, generation_id: int) -> bool:
        return (self._pending_clear is not None
                and self._pending_clear.generation_id == generation_id)

    def resolve_clear(self, generation_id: int) -> GenerationToken | None:
        if not self.accepts_clear(generation_id):
            return None
        token = self._pending_clear
        self._pending_clear = None
        return token

    def finish(self, token: GenerationToken) -> None:
        if self._current == token:
            self._current = None
            self._task = None

    async def close(self) -> None:
        self._pending_clear = None
        await self.cancel_current()
