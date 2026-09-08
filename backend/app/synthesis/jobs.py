"""Owned synthesis tasks: duplicate submissions observe one admitted worker."""

import asyncio
import json
import logging

from app.inference.base import InputTooLongError, UnsupportedVoiceError
from app.models.loader import ModelLoadError
from app.models.registry import ModelNotFoundError
from app.models.resources import ResourceUnavailableError
from app.schemas.synthesis import SynthesisRequest
from app.schemas.synthesis_job import SynthesisJob, SynthesisJobError
from app.synthesis.job_store import JobStore, SynthesisJobCapacityError
from app.synthesis.service import SynthesisService
from app.text_processing.service import TextProcessingError

LOGGER = logging.getLogger("uvicorn.error")


class SynthesisJobSubmissionError(RuntimeError):
    def __init__(self, error: SynthesisJobError):
        super().__init__(error.detail)
        self.status_code = error.status_code


def _public_error(error: Exception) -> SynthesisJobError:
    if isinstance(error, ModelNotFoundError):
        code = 404
    elif isinstance(error, (InputTooLongError, UnsupportedVoiceError, TextProcessingError)):
        code = 422
    elif isinstance(error, (ResourceUnavailableError, ModelLoadError)):
        code = 503
    else:
        code = 500
    return SynthesisJobError(status_code=code, detail=str(error))


class SynthesisJobService:
    """One service per event loop/process; no queue or automatic inference retries."""

    def __init__(self, synthesis: SynthesisService, store: JobStore) -> None:
        self._synthesis = synthesis
        self._store = store
        self._tasks: set[asyncio.Task[None]] = set()
        self._admissions: dict[str, asyncio.Future[None]] = {}
        self._closing = False

    def _log(self, event: str, job: SynthesisJob) -> None:
        fields = {
            "event": event,
            "job_id": job.job_id,
            "model_id": job.request.model_id,
            "status": job.status,
        }
        LOGGER.info(json.dumps(fields), extra=fields)

    async def start(self, request: SynthesisRequest, key: str) -> SynthesisJob:
        if self._closing:
            raise SynthesisJobCapacityError("Synthesis service is shutting down.")
        # No awaits between the atomic claim and task registration. A duplicate
        # can never observe a claimed job without its admission waiter.
        job, created = self._store.claim(key, request)
        identifier = job.job_id
        if created:
            admission = asyncio.get_running_loop().create_future()
            self._admissions[identifier] = admission
            coroutine = self._execute(job, admission)
            try:
                task = asyncio.create_task(coroutine)
            except Exception as error:
                coroutine.close()
                self._admissions.pop(identifier)
                self._store.update(identifier, status="rejected", error=_public_error(error))
                raise SynthesisJobSubmissionError(_public_error(error)) from error
            self._tasks.add(task)

            def finished(task: asyncio.Task[None]) -> None:
                self._tasks.discard(task)
                self._admissions.pop(identifier, None)

            task.add_done_callback(finished)
        else:
            self._log("synthesis_job_replayed", job)
        admission = self._admissions.get(identifier)
        if admission is not None:
            # The HTTP request owns only observation, never the worker or waiter.
            await asyncio.shield(admission)
        job = self._store.get(identifier)
        if job.status == "rejected":
            assert job.error is not None
            raise SynthesisJobSubmissionError(job.error)
        return job

    def get(self, job_id: str) -> SynthesisJob:
        return self._store.get(job_id)

    async def _execute(self, job: SynthesisJob, admission: asyncio.Future[None]) -> None:
        loop = asyncio.get_running_loop()
        identifier = job.job_id

        def admitted() -> None:
            updated = self._store.update(identifier, status="loading")
            self._log("synthesis_job_admitted", updated)
            admission.set_result(None)

        def stage(value: str) -> None:
            self._store.update(identifier, status=value)

        try:
            coroutine = self._synthesis.synthesize(
                job.request,
                on_admitted=lambda: loop.call_soon_threadsafe(admitted),
                on_stage=lambda value: loop.call_soon_threadsafe(stage, value),
            )
            try:
                worker = asyncio.create_task(coroutine)
            except Exception:
                coroutine.close()
                raise
            # A cancellation of the supervising task must not abandon a native
            # inference thread or publish completion before its lease is released.
            while True:
                try:
                    result = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    if worker.cancelled():
                        # Event-loop termination can also cancel the child task.
                        # The process-local record cannot be recovered after exit.
                        raise
            updated = self._store.update(identifier, status="completed", result=result)
        except Exception as error:  # noqa: BLE001 - persist failures from every synthesis boundary.
            updated = self._store.update(
                identifier,
                status="failed" if admission.done() else "rejected",
                error=_public_error(error),
            )
        finally:
            if not admission.done():
                admission.set_result(None)
        self._log("synthesis_job_finished", updated)

    async def close(self) -> None:
        """Drain owned workers during graceful shutdown, including WAV storage."""
        self._closing = True
        if self._tasks:
            await asyncio.shield(asyncio.gather(*tuple(self._tasks)))
