"""Bounded event-driven dispatcher shared by backend compute workloads."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from app.resources.manager import ResourceManager
from app.resources.profiles import ResourceDemand
from app.scheduling.policy import select_next

SchedulerState = Literal["queued", "reserved", "running"]
StateCallback = Callable[[SchedulerState, str | None], None]
CompletionCallback = Callable[[object | None, BaseException | None, bool], None]


class ProcessingQueueFullError(RuntimeError):
    """Raised when the bounded pending queue cannot accept another job."""


class ResourceUnavailableError(RuntimeError):
    """Raised when work cannot fit even on an otherwise idle instance."""


class ProcessingJobNotFoundError(LookupError):
    """Raised when cancellation targets an unknown scheduler job."""


class ProcessingJobCancelledError(RuntimeError):
    """Raised to a waiter after accepted work is cancelled."""


class ProcessingJobExpiredError(RuntimeError):
    """Raised when pending work exceeds the configured queue wait limit."""


@dataclass(slots=True)
class ScheduledWork:
    id: str
    demand: ResourceDemand
    work: Callable[[], object]
    payload_bytes: int = 0
    on_state: StateCallback | None = None
    on_complete: CompletionCallback | None = None
    enqueued_at: float = 0.0
    sequence: int = 0
    future: asyncio.Future[Any] | None = None
    cancellation_requested: bool = False


class ProcessingScheduler:
    """Dispatch fitting work and reserve the next start after the aging deadline."""

    def __init__(
        self,
        resource_manager: ResourceManager,
        *,
        aging_threshold_seconds: float = 30.0,
        queue_capacity: int = 32,
        maximum_payload_bytes: int = 160_000,
        sample_interval_seconds: float = 1.0,
        queue_wait_timeout_seconds: float = 300.0,
        maximum_workers: int = 4,
        clock=time.monotonic,
    ) -> None:
        if aging_threshold_seconds < 0:
            raise ValueError("aging_threshold_seconds cannot be negative")
        if queue_capacity < 1 or maximum_payload_bytes < 1:
            raise ValueError("queue limits must be positive")
        if sample_interval_seconds <= 0 or queue_wait_timeout_seconds <= 0:
            raise ValueError("scheduler timeouts must be positive")
        if maximum_workers < 1:
            raise ValueError("maximum_workers must be at least one")
        self._resources = resource_manager
        self._aging_threshold = aging_threshold_seconds
        self._queue_capacity = queue_capacity
        self._maximum_payload_bytes = maximum_payload_bytes
        self._sample_interval = sample_interval_seconds
        self._queue_wait_timeout = queue_wait_timeout_seconds
        self._clock = clock
        self._pending: list[ScheduledWork] = []
        self._active: dict[str, ScheduledWork] = {}
        self._sequence = 0
        self._pending_bytes = 0
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: asyncio.Task[None] | None = None
        self._workers: set[asyncio.Task[None]] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="openvoice-work",
        )
        self._closed = False

    async def submit(self, item: ScheduledWork) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._pending or self._active:
                raise RuntimeError("The processing scheduler cannot move between active loops.")
            self._loop = loop
            self._lock = asyncio.Lock()
            self._wake = asyncio.Event()
            self._runner = None
        async with self._lock:
            if self._closed:
                raise RuntimeError("The processing scheduler is shutting down.")
            if len(self._pending) >= self._queue_capacity:
                raise ProcessingQueueFullError(
                    "The processing queue is full. Retry after an active job finishes."
                )
            if self._pending_bytes + item.payload_bytes > self._maximum_payload_bytes:
                raise ProcessingQueueFullError(
                    "The processing queue text budget is full. Retry after queued work starts."
                )
            self._resources.refresh()
            if not self._resources.can_ever_fit(item.demand).fits:
                raise ResourceUnavailableError(
                    "This request cannot fit within the instance CPU or memory budget."
                )
            self._sequence += 1
            item.sequence = self._sequence
            item.enqueued_at = self._clock()
            item.future = loop.create_future()
            self._pending.append(item)
            self._pending_bytes += item.payload_bytes
            self._set_state(item, "queued", self._resources.can_admit(item.demand).reason)
            self._wake.set()
            if self._runner is None or self._runner.done():
                self._runner = asyncio.create_task(self._dispatch_loop())
            return item.future

    async def run(
        self,
        *,
        job_id: str,
        demand: ResourceDemand,
        work: Callable[[], object],
        payload_bytes: int = 0,
        on_state: StateCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> object:
        future = await self.submit(
            ScheduledWork(
                id=job_id,
                demand=demand,
                work=work,
                payload_bytes=payload_bytes,
                on_state=on_state,
                on_complete=on_complete,
            )
        )
        return await future

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            for index, item in enumerate(self._pending):
                if item.id != job_id:
                    continue
                self._pending.pop(index)
                self._pending_bytes -= item.payload_bytes
                item.cancellation_requested = True
                error = ProcessingJobCancelledError("The queued request was cancelled.")
                if item.future is not None and not item.future.done():
                    item.future.set_exception(error)
                self._complete(item, None, error, True)
                self._wake.set()
                return True
            active = self._active.get(job_id)
            if active is not None:
                active.cancellation_requested = True
                self._wake.set()
                return False
        raise ProcessingJobNotFoundError(f"Processing job '{job_id}' was not found.")

    def status(self) -> dict[str, object]:
        resource_status = self._resources.status()
        resource_status.update(
            queuedJobs=len(self._pending),
            activeJobIds=list(self._active),
            agingThresholdSeconds=self._aging_threshold,
        )
        return resource_status

    async def close(self, grace_seconds: float = 10.0) -> None:
        async with self._lock:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._pending_bytes = 0
            for item in pending:
                error = ProcessingJobCancelledError("Server shutdown cancelled queued work.")
                if item.future is not None and not item.future.done():
                    item.future.set_exception(error)
                self._complete(item, None, error, True)
            self._wake.set()
        if self._workers:
            done, _pending = await asyncio.wait(self._workers, timeout=grace_seconds)
            for task in done:
                task.result()
        self._executor.shutdown(wait=False, cancel_futures=False)

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                dispatch: list[ScheduledWork] = []
                wait_for = self._sample_interval
                async with self._lock:
                    self._wake.clear()
                    self._resources.refresh()
                    self._expire_pending()
                    while self._pending:
                        now = self._clock()
                        by_id = {item.id: item for item in self._pending}
                        selection = select_next(
                            self._pending,
                            now=now,
                            aging_threshold_seconds=self._aging_threshold,
                            fits=lambda item: self._resources.can_admit(item.demand).fits,
                        )
                        if selection.selected_id is None:
                            self._mark_waiters(selection.reserved_id)
                            break
                        selected = by_id[selection.selected_id]
                        decision = self._resources.can_admit(selected.demand)
                        if not decision.fits:
                            self._mark_waiters(selection.reserved_id)
                            break
                        self._resources.reserve(selected.id, selected.demand)
                        self._pending.remove(selected)
                        self._pending_bytes -= selected.payload_bytes
                        self._active[selected.id] = selected
                        dispatch.append(selected)
                        if selection.reserved_id is not None:
                            self._set_state(selected, "reserved", None)

                    for item in dispatch:
                        task = asyncio.create_task(self._execute(item))
                        self._workers.add(task)
                        task.add_done_callback(self._workers.discard)

                    if not self._pending and not self._active:
                        return
                    if self._pending:
                        oldest = min(item.enqueued_at for item in self._pending)
                        deadline = oldest + self._aging_threshold
                        wait_for = min(wait_for, max(0.001, deadline - self._clock()))
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=wait_for)
                except TimeoutError:
                    pass
        finally:
            self._runner = None

    def _mark_waiters(self, reserved_id: str | None) -> None:
        for item in self._pending:
            if reserved_id == item.id:
                reason = self._resources.can_admit(item.demand).reason
                self._set_state(item, "reserved", reason)
            elif reserved_id is not None:
                self._set_state(item, "queued", "older-reserved-request")
            else:
                reason = self._resources.can_admit(item.demand).reason
                self._set_state(item, "queued", reason)

    def _expire_pending(self) -> None:
        now = self._clock()
        expired = [
            item
            for item in self._pending
            if now - item.enqueued_at >= self._queue_wait_timeout
        ]
        for item in expired:
            self._pending.remove(item)
            self._pending_bytes -= item.payload_bytes
            error = ProcessingJobExpiredError(
                "The request expired before enough compute resources became available."
            )
            if item.future is not None and not item.future.done():
                item.future.set_exception(error)
            self._complete(item, None, error, False)

    async def _execute(self, item: ScheduledWork) -> None:
        result: object | None = None
        error: BaseException | None = None
        try:
            self._set_state(item, "running", None)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, item.work)
        except BaseException as caught:  # noqa: BLE001 - preserve worker failure.
            error = caught
        finally:
            async with self._lock:
                self._resources.release(item.id)
                self._active.pop(item.id, None)
                self._wake.set()
        if item.cancellation_requested and error is None:
            error = ProcessingJobCancelledError(
                "Cancellation completed after active inference stopped."
            )
        self._complete(item, result, error, item.cancellation_requested)
        if item.future is not None and not item.future.done():
            if error is not None:
                item.future.set_exception(error)
            else:
                item.future.set_result(result)

    @staticmethod
    def _set_state(item: ScheduledWork, state: SchedulerState, reason: str | None) -> None:
        if item.on_state is not None:
            item.on_state(state, reason)

    @staticmethod
    def _complete(
        item: ScheduledWork,
        result: object | None,
        error: BaseException | None,
        cancelled: bool,
    ) -> None:
        if item.on_complete is not None:
            item.on_complete(result, error, cancelled)
