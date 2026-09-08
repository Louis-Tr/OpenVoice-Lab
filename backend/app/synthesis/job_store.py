"""Bounded, process-local job identity and snapshots with atomic key claims."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Lock
from typing import Protocol
from uuid import uuid4

from app.schemas.synthesis import SynthesisRequest
from app.schemas.synthesis_job import SynthesisJob

TERMINAL_STATUSES = {"completed", "failed", "rejected"}


class SynthesisJobNotFoundError(LookupError):
    """Unknown or expired job in this serving process."""


class IdempotencyConflictError(ValueError):
    """A retained key cannot be bound to a different validated request."""


class SynthesisJobCapacityError(RuntimeError):
    """The job store cannot accept more identities without breaking retention."""


class JobStore(Protocol):
    def claim(self, key: str, request: SynthesisRequest) -> tuple[SynthesisJob, bool]: ...
    def get(self, job_id: str) -> SynthesisJob: ...
    def update(self, job_id: str, **changes: object) -> SynthesisJob: ...


@dataclass
class _Entry:
    key: str
    fingerprint: str
    job: SynthesisJob


class InMemoryJobStore:
    """Expire terminal records only; keys and results share the same lifetime."""

    def __init__(
        self,
        *,
        maximum_jobs: int = 1000,
        retention_seconds: float = 86400,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if maximum_jobs < 1 or retention_seconds <= 0:
            raise ValueError("Job capacity and retention must be positive.")
        self._maximum = maximum_jobs
        self._retention = timedelta(seconds=retention_seconds)
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._keys: dict[str, str] = {}
        self._lock = Lock()

    def _prune(self, now: datetime) -> None:
        expired = [
            identifier
            for identifier, entry in self._entries.items()
            if entry.job.expires_at is not None and entry.job.expires_at <= now
        ]
        for identifier in expired:
            entry = self._entries.pop(identifier)
            del self._keys[entry.key]

    def claim(self, key: str, request: SynthesisRequest) -> tuple[SynthesisJob, bool]:
        canonical = json.dumps(
            request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            now = self._clock()
            self._prune(now)
            if key in self._keys:
                entry = self._entries[self._keys[key]]
                if entry.fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "Idempotency-Key was already used with a different synthesis request."
                    )
                return entry.job.model_copy(deep=True), False
            if len(self._entries) >= self._maximum:
                raise SynthesisJobCapacityError("Synthesis job store is full; retry later.")
            job = SynthesisJob(
                job_id=uuid4().hex,
                request=request.model_copy(deep=True),
                created_at=now,
                updated_at=now,
            )
            self._entries[job.job_id] = _Entry(key, fingerprint, job)
            self._keys[key] = job.job_id
            return job.model_copy(deep=True), True

    def get(self, job_id: str) -> SynthesisJob:
        with self._lock:
            self._prune(self._clock())
            if job_id not in self._entries:
                raise SynthesisJobNotFoundError("Synthesis job was not found or has expired.")
            return self._entries[job_id].job.model_copy(deep=True)

    def update(self, job_id: str, **changes: object) -> SynthesisJob:
        with self._lock:
            entry = self._entries[job_id]
            if entry.job.status in TERMINAL_STATUSES:
                return entry.job.model_copy(deep=True)
            now = self._clock()
            changes["updated_at"] = now
            if changes.get("status") in TERMINAL_STATUSES:
                changes.update(completed_at=now, expires_at=now + self._retention)
            entry.job = entry.job.model_copy(update=changes, deep=True)
            return entry.job.model_copy(deep=True)
