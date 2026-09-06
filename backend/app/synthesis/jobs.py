"""Synthesis-specific job state layered over the shared scheduler."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from uuid import uuid4

from app.resources.profiles import ResourceProfileRegistry
from app.scheduling.service import (
    ProcessingJobCancelledError,
    ProcessingJobNotFoundError,
    ProcessingScheduler,
    ScheduledWork,
    SchedulerState,
)
from app.schemas.jobs import SynthesisJobError, SynthesisJobState, SynthesisJobStatus
from app.schemas.synthesis import SynthesisRequest, SynthesisResult
from app.synthesis.service import PreparedSynthesis, SynthesisService


class SynthesisJobNotFoundError(LookupError):
    """Raised when a synthesis job is not retained in this process."""


class IdempotencyConflictError(RuntimeError):
    """Raised when one client reuses a key with a different request."""


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class _StoredJob:
    status: SynthesisJobStatus
    prepared: PreparedSynthesis
    payload_hash: str
    client_scope: str
    idempotency_key: str | None
    future: asyncio.Future[object] | None = None


class SynthesisJobService:
    """Validate, retain, inspect, and cancel queued synthesis requests."""

    def __init__(
        self,
        synthesis: SynthesisService,
        scheduler: ProcessingScheduler,
        profiles: ResourceProfileRegistry,
        *,
        retention_seconds: float = 3600.0,
        maximum_retained_jobs: int = 200,
    ) -> None:
        self._synthesis = synthesis
        self._scheduler = scheduler
        self._profiles = profiles
        self._retention = timedelta(seconds=retention_seconds)
        self._maximum_retained = maximum_retained_jobs
        self._jobs: dict[str, _StoredJob] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._guard = Lock()

    async def submit(
        self,
        request: SynthesisRequest,
        *,
        client_scope: str = "anonymous",
        idempotency_key: str | None = None,
    ) -> SynthesisJobStatus:
        prepared = self._synthesis.prepare(request)
        payload = json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
        payload_hash = hashlib.sha256(payload).hexdigest()
        with self._guard:
            self._prune_locked()
            if idempotency_key:
                existing_id = self._idempotency.get((client_scope, idempotency_key))
                if existing_id is not None:
                    existing = self._jobs.get(existing_id)
                    if existing is not None and existing.payload_hash == payload_hash:
                        return existing.status.model_copy(deep=True)
                    raise IdempotencyConflictError(
                        "This idempotency key was already used for a different request."
                    )
            now = utc_now()
            identifier = f"synthesis-{uuid4().hex}"
            status = SynthesisJobStatus(
                id=identifier,
                state="queued",
                request=request,
                normalized_text=prepared.normalized_text,
                queued_at=now,
                updated_at=now,
            )
            stored = _StoredJob(
                status=status,
                prepared=prepared,
                payload_hash=payload_hash,
                client_scope=client_scope,
                idempotency_key=idempotency_key,
            )
            self._jobs[identifier] = stored
            if idempotency_key:
                self._idempotency[(client_scope, idempotency_key)] = identifier

        demand = self._profiles.for_model(
            prepared.model,
            loaded=self._synthesis.model_loader.is_loaded(prepared.model.model_id),
            input_characters=len(prepared.normalized_text),
        )
        item = ScheduledWork(
            id=identifier,
            demand=demand,
            payload_bytes=len(request.text.encode("utf-8")),
            work=lambda: self._synthesis.execute(
                prepared,
                lambda state: self._set_processing_state(identifier, state),
            ),
            on_state=lambda state, reason: self._set_scheduler_state(
                identifier, state, reason
            ),
            on_complete=lambda result, error, cancelled: self._finish(
                identifier, result, error, cancelled
            ),
        )
        try:
            future = await self._scheduler.submit(item)
        except Exception:
            with self._guard:
                self._delete_locked(identifier)
            raise
        future.add_done_callback(self._consume_background_failure)
        with self._guard:
            stored.future = future
            return stored.status.model_copy(deep=True)

    async def submit_and_wait(
        self,
        request: SynthesisRequest,
        disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> SynthesisResult:
        job = await self.submit(request)
        with self._guard:
            future = self._jobs[job.id].future
        if future is None:
            raise RuntimeError("The accepted synthesis job has no completion future.")
        while not future.done():
            await asyncio.wait((future,), timeout=0.25)
            if disconnected is not None and not future.done() and await disconnected():
                await self.cancel(job.id)
        result = await future
        if not isinstance(result, SynthesisResult):
            raise TypeError("The synthesis worker returned an invalid result.")
        return result

    def get(self, job_id: str) -> SynthesisJobStatus:
        with self._guard:
            self._prune_locked()
            stored = self._jobs.get(job_id)
            if stored is None:
                raise SynthesisJobNotFoundError(f"Synthesis job '{job_id}' was not found.")
            return stored.status.model_copy(deep=True)

    async def cancel(self, job_id: str) -> SynthesisJobStatus:
        with self._guard:
            stored = self._jobs.get(job_id)
            if stored is None:
                raise SynthesisJobNotFoundError(f"Synthesis job '{job_id}' was not found.")
            if stored.status.state in {"completed", "failed", "cancelled"}:
                return stored.status.model_copy(deep=True)
            stored.status = stored.status.model_copy(
                update={"cancellation_requested": True, "updated_at": utc_now()}
            )
        try:
            await self._scheduler.cancel(job_id)
        except ProcessingJobNotFoundError:
            pass
        return self.get(job_id)

    def _set_scheduler_state(
        self,
        job_id: str,
        state: SchedulerState,
        reason: str | None,
    ) -> None:
        now = utc_now()
        mapped: SynthesisJobState = "loading" if state == "running" else state
        with self._guard:
            stored = self._jobs.get(job_id)
            if stored is None or stored.status.state in {"completed", "failed", "cancelled"}:
                return
            changes: dict[str, object] = {
                "state": mapped,
                "waiting_reason": reason,
                "updated_at": now,
            }
            if state == "running" and stored.status.started_at is None:
                changes["started_at"] = now
                changes["queue_wait_ms"] = (
                    now - stored.status.queued_at
                ).total_seconds() * 1000
            stored.status = stored.status.model_copy(update=changes)

    def _set_processing_state(self, job_id: str, state: str) -> None:
        if state not in {"loading", "running", "saving"}:
            return
        now = utc_now()
        with self._guard:
            stored = self._jobs.get(job_id)
            if stored is None:
                return
            changes: dict[str, object] = {
                "state": state,
                "waiting_reason": None,
                "updated_at": now,
            }
            if stored.status.started_at is None:
                changes["started_at"] = now
                changes["queue_wait_ms"] = (
                    now - stored.status.queued_at
                ).total_seconds() * 1000
            stored.status = stored.status.model_copy(update=changes)

    def _finish(
        self,
        job_id: str,
        result: object | None,
        error: BaseException | None,
        cancelled: bool,
    ) -> None:
        now = utc_now()
        with self._guard:
            stored = self._jobs.get(job_id)
            if stored is None:
                return
            processing_ms = None
            if stored.status.started_at is not None:
                processing_ms = (now - stored.status.started_at).total_seconds() * 1000
            if cancelled or isinstance(error, ProcessingJobCancelledError):
                changes: dict[str, object] = {
                    "state": "cancelled",
                    "completed_at": now,
                    "updated_at": now,
                    "processing_ms": processing_ms,
                    "waiting_reason": None,
                }
            elif error is not None:
                changes = {
                    "state": "failed",
                    "completed_at": now,
                    "updated_at": now,
                    "processing_ms": processing_ms,
                    "waiting_reason": None,
                    "error": SynthesisJobError(
                        code=type(error).__name__,
                        message=str(error),
                    ),
                }
            else:
                changes = {
                    "state": "completed",
                    "completed_at": now,
                    "updated_at": now,
                    "processing_ms": processing_ms,
                    "waiting_reason": None,
                    "result": result if isinstance(result, SynthesisResult) else None,
                }
            stored.status = stored.status.model_copy(update=changes)

    def _prune_locked(self) -> None:
        cutoff = utc_now() - self._retention
        terminal = [
            job
            for job in self._jobs.values()
            if job.status.completed_at is not None
        ]
        expired = {
            job.status.id
            for job in terminal
            if job.status.completed_at is not None and job.status.completed_at < cutoff
        }
        if len(self._jobs) - len(expired) > self._maximum_retained:
            ordered = sorted(terminal, key=lambda job: job.status.completed_at or utc_now())
            excess = len(self._jobs) - len(expired) - self._maximum_retained
            expired.update(job.status.id for job in ordered[:excess])
        for identifier in expired:
            self._delete_locked(identifier)

    def _delete_locked(self, identifier: str) -> None:
        stored = self._jobs.pop(identifier, None)
        if stored is not None and stored.idempotency_key:
            self._idempotency.pop((stored.client_scope, stored.idempotency_key), None)

    @staticmethod
    def _consume_background_failure(future: asyncio.Future[object]) -> None:
        if not future.cancelled():
            future.exception()
