"""Idempotency, admission, refresh recovery and thread lifetime without real models."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from threading import Barrier, Event
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from test_model_loader import CloseableEngine, definition

from app.api.errors import register_error_handlers
from app.api.synthesis import create_router
from app.audio.service import AudioService
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelRegistry
from app.models.resources import ResourceManager
from app.models.scheduler import EngineScheduler
from app.schemas.synthesis import SynthesisRequest
from app.synthesis.job_store import (
    IdempotencyConflictError,
    InMemoryJobStore,
    SynthesisJobCapacityError,
    SynthesisJobNotFoundError,
)
from app.synthesis.jobs import SynthesisJobService, SynthesisJobSubmissionError
from app.synthesis.service import SynthesisService
from app.text_processing.service import TextProcessingService

PAYLOAD = {"text": "Hello", "modelId": "a", "voiceId": "voice"}
URL = "/api/synthesis/jobs"


@asynccontextmanager
async def harness(tmp_path, *, failure=None, units=2, memory=10_000, maximum_jobs=1000):
    gates = {stage: Event() for stage in ("admission", "loading", "generating", "saving")}
    entered = {stage: Event() for stage in gates}
    calls = {"loading": 0, "generating": 0, "saving": 0}
    for gate in gates.values():
        gate.set()

    def visit(stage):
        if stage in calls:
            calls[stage] += 1
        entered[stage].set()
        assert gates[stage].wait(10), f"Timed out at {stage}"
        if stage == failure:
            raise RuntimeError(f"{stage} failed")

    class Engine(CloseableEngine):
        def synthesize(self, *args, **kwargs):
            visit("generating")
            return super().synthesize(*args, **kwargs)

    def construct(model):
        visit("loading")
        return Engine(model.key)

    class Scheduler(EngineScheduler):
        def _reserve(self, model):
            visit("admission")
            return super()._reserve(model)

    class Audio(AudioService):
        def create_artifact(self, *args, **kwargs):
            visit("saving")
            assert resources.reserved_cpu_units > 0
            return super().create_artifact(*args, **kwargs)

    registry = ModelRegistry([definition(key, tmp_path / key) for key in ("a", "b")])
    resources = ResourceManager(
        profiles={"a": 100, "b": 100},
        cpu_units=units,
        cpu_threads=1,
        memory_reader=lambda: memory,
        memory_headroom_mb=0,
    )
    loader = ModelLoader(engine_factory=construct)
    metrics = MetricsCollector()
    scheduler = Scheduler(loader, resources, metrics)
    audio = Audio(tmp_path / "audio")
    synthesis = SynthesisService(registry, scheduler, audio, metrics, TextProcessingService())
    store = InMemoryJobStore(maximum_jobs=maximum_jobs)
    jobs = SynthesisJobService(synthesis, store)
    app = FastAPI()
    app.include_router(create_router(synthesis, jobs), prefix="/api")
    app.mount("/audio", StaticFiles(directory=audio.output_dir))
    register_error_handlers(app)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            yield SimpleNamespace(
                client=client,
                transport=transport,
                jobs=jobs,
                scheduler=scheduler,
                resources=resources,
                loader=loader,
                calls=calls,
                gates=gates,
                entered=entered,
            )
        finally:
            for gate in gates.values():
                gate.set()
            await jobs.close()


async def submit(h, key="one", payload=None):
    return await h.client.post(URL, json=payload or PAYLOAD, headers={"Idempotency-Key": key})


async def wait_for_stage(h, stage):
    assert await asyncio.to_thread(h.entered[stage].wait, 5)
    # Thread notifications are posted before the corresponding blocking operation.
    await asyncio.sleep(0)


async def drain(h):
    await asyncio.wait_for(asyncio.gather(*tuple(h.jobs._tasks)), timeout=5)


def test_simultaneous_duplicates_return_before_loading_and_share_one_generation(tmp_path):
    async def scenario():
        async with harness(tmp_path) as h:
            for stage in ("loading", "generating", "saving"):
                h.gates[stage].clear()
            responses = await asyncio.wait_for(
                asyncio.gather(*(submit(h) for _ in range(8))),
                timeout=5,
            )
            assert {r.status_code for r in responses} == {202}
            assert len({r.json()["jobId"] for r in responses}) == 1
            location = responses[0].headers["location"]
            assert responses[0].headers["retry-after"] == "2"
            assert responses[0].headers["cache-control"] == "no-store"
            await wait_for_stage(h, "loading")
            assert h.calls["loading"] == 1
            assert (await h.client.get(location)).json()["status"] == "loading"
            assert h.resources.reserved_cpu_units == 1
            # Different keys do not bypass the pending-model admission rule.
            assert (await submit(h, "other")).status_code == 503
            h.gates["loading"].set()
            await wait_for_stage(h, "generating")
            assert (await h.client.get(location)).json()["status"] == "generating"
            h.gates["generating"].set()
            await wait_for_stage(h, "saving")
            assert (await h.client.get(location)).json()["status"] == "saving"
            assert h.resources.reserved_cpu_units == 1
            h.gates["saving"].set()
            await drain(h)
            result = (await h.client.get(location)).json()
            assert result["status"] == "completed"
            assert result["expiresAt"] is not None
            replay = await submit(h)
            assert replay.status_code == 200
            assert replay.json() == result
            audio = await h.client.get(result["result"]["audioUrl"])
            assert audio.content.startswith(b"RIFF")
            assert h.calls == {"loading": 1, "generating": 1, "saving": 1}
            assert h.resources.reserved_cpu_units == 0
            assert h.resources.reserved_memory_mb == 0
            assert h.loader.load_count("a") == 1

    asyncio.run(scenario())


def test_lost_post_response_and_refresh_do_not_cancel_or_restart_work(tmp_path):
    async def scenario():
        async with harness(tmp_path) as h:
            h.gates["admission"].clear()
            h.gates["generating"].clear()
            first = asyncio.create_task(submit(h))
            await wait_for_stage(h, "admission")
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            retry = asyncio.create_task(submit(h))
            h.gates["admission"].set()
            response = await asyncio.wait_for(retry, 5)
            await wait_for_stage(h, "generating")
            async with httpx.AsyncClient(
                transport=h.transport, base_url="http://test"
            ) as refreshed:
                restored = await refreshed.get(response.headers["location"])
            assert restored.json()["status"] == "generating"
            assert h.calls["generating"] == 1
            assert h.resources.reserved_cpu_units == 1
            closing = asyncio.create_task(h.jobs.close())
            await asyncio.sleep(0)
            assert not closing.done()
            assert h.resources.reserved_cpu_units == 1
            assert (await submit(h, "new")).status_code == 503
            h.gates["generating"].set()
            await asyncio.wait_for(closing, 5)
            assert h.resources.reserved_cpu_units == 0
            assert h.jobs.get(response.json()["jobId"]).status == "completed"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["loading", "generating", "saving"])
def test_failure_is_retained_and_reservations_are_released(tmp_path, phase):
    async def scenario():
        async with harness(tmp_path, failure=phase) as h:
            first = await submit(h)
            assert first.status_code in (200, 202)
            await drain(h)
            job = (await h.client.get(first.headers["location"])).json()
            assert job["status"] == "failed"
            assert phase in job["error"]["detail"]
            assert job["error"]["statusCode"] == (503 if phase == "loading" else 500)
            before = dict(h.calls)
            replay = await submit(h)
            assert replay.status_code == 200
            assert replay.json() == job
            assert h.calls == before
            assert h.resources.reserved_cpu_units == 0
            assert h.resources.reserved_memory_mb == 0
            assert not h.scheduler._loading
            assert ("a" in h.scheduler._records) == (phase != "loading")

    asyncio.run(scenario())


@pytest.mark.parametrize("units,memory", [(1, 10_000), (2, 150)])
def test_capacity_rejection_is_immediate_and_replayed(tmp_path, units, memory):
    async def scenario():
        async with harness(tmp_path, units=units, memory=memory) as h:
            h.gates["generating"].clear()
            first = await submit(h)
            await wait_for_stage(h, "generating")
            second = await submit(h, "two", {**PAYLOAD, "modelId": "b"})
            assert second.status_code == 503
            assert "detail" in second.json()
            # A replay of an admitted job consumes no additional CPU or memory.
            assert (await submit(h)).json()["jobId"] == first.json()["jobId"]
            assert h.resources.reserved_cpu_units == 1
            h.gates["generating"].set()
            await drain(h)
            replay = await submit(h, "two", {**PAYLOAD, "modelId": "b"})
            assert replay.status_code == 503
            assert replay.json() == second.json()
            assert h.calls["loading"] == 1
            # A deliberate retry uses a new key after resources become free.
            assert (await submit(h, "three", {**PAYLOAD, "modelId": "b"})).status_code in (200, 202)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change,code",
    [
        ({"modelId": "missing"}, 404),
        ({"voiceId": "missing"}, 422),
        ({"text": "!!!"}, 422),
        ({"text": "x" * 5001}, 422),
    ],
)
def test_invalid_request_does_not_load_an_engine(tmp_path, change, code):
    async def scenario():
        async with harness(tmp_path) as h:
            response = await submit(h, payload={**PAYLOAD, **change})
            assert response.status_code == code
            assert "detail" in response.json()
            assert not any(h.calls.values())
            assert h.resources.reserved_cpu_units == 0

    asyncio.run(scenario())


def test_header_validation_conflicts_missing_jobs_and_store_limit(tmp_path):
    async def scenario():
        async with harness(tmp_path, maximum_jobs=1) as h:
            for key in (None, "", "has spaces", "x" * 129):
                headers = {} if key is None else {"Idempotency-Key": key}
                response = await h.client.post(URL, json=PAYLOAD, headers=headers)
                assert response.status_code == 422
            assert (await h.client.get(URL + "/missing")).status_code == 404
            original = await submit(h)
            await drain(h)
            conflict = await submit(h, payload={**PAYLOAD, "text": "Different"})
            assert conflict.status_code == 409
            assert "detail" in conflict.json()
            assert (await submit(h, "new")).status_code == 503
            assert (await submit(h)).json()["jobId"] == original.json()["jobId"]
            assert h.calls["generating"] == 1

    asyncio.run(scenario())


def test_independent_keys_still_run_concurrently_on_shared_engine(tmp_path):
    async def scenario():
        async with harness(tmp_path) as h:
            # Warm the engine without creating a job or running inference.
            model = h.jobs._synthesis._model_registry.get("a")
            with h.scheduler.acquire(model):
                pass
            h.gates["generating"].clear()
            first, second = await asyncio.gather(submit(h, "one"), submit(h, "two"))
            assert first.json()["jobId"] != second.json()["jobId"]
            assert first.status_code == second.status_code == 202
            assert h.resources.reserved_cpu_units == 2
            h.gates["generating"].set()
            await drain(h)
            assert h.calls["generating"] == 2
            assert h.loader.load_count("a") == 1

    asyncio.run(scenario())


def test_store_atomic_claims_defaults_and_detached_snapshots():
    store = InMemoryJobStore()
    barrier = Barrier(8)
    request = SynthesisRequest.model_validate(PAYLOAD)

    def claim():
        barrier.wait(5)
        return store.claim("same", request)

    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(lambda _: claim(), range(8)))
    assert sum(created for _, created in outcomes) == 1
    assert len({job.job_id for job, _ in outcomes}) == 1
    original = outcomes[0][0]
    original.request.text = "mutated"
    assert store.get(original.job_id).request.text == "Hello"
    explicit = SynthesisRequest.model_validate(
        {**PAYLOAD, "sanitizeText": True, "normalizeText": True}
    )
    assert store.claim("same", explicit)[1] is False
    for change in (
        {"text": "changed"},
        {"model_id": "b"},
        {"voice_id": "other"},
        {"sanitize_text": False},
        {"normalize_text": False},
    ):
        with pytest.raises(IdempotencyConflictError):
            store.claim("same", request.model_copy(update=change))


def test_retention_expires_terminal_keys_together_without_evicting_active_jobs():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemoryJobStore(maximum_jobs=2, retention_seconds=10, clock=lambda: now)
    request = SynthesisRequest.model_validate(PAYLOAD)
    terminal, _ = store.claim("terminal", request)
    active, _ = store.claim("active", request)
    store.update(terminal.job_id, status="failed")
    with pytest.raises(SynthesisJobCapacityError):
        store.claim("third", request)
    now += timedelta(seconds=11)
    with pytest.raises(SynthesisJobNotFoundError):
        store.get(terminal.job_id)
    assert store.get(active.job_id).status == "pending"
    replacement, created = store.claim("terminal", request)
    assert created and replacement.job_id != terminal.job_id


def test_admission_callback_failure_releases_pending_reservation(tmp_path):
    async def scenario():
        async with harness(tmp_path) as h:

            def fail():
                raise RuntimeError("observer failed")

            with (
                pytest.raises(RuntimeError, match="observer failed"),
                h.scheduler.acquire(definition("a", tmp_path / "a"), on_admitted=fail),
            ):
                pass
            assert h.resources.reserved_cpu_units == 0
            assert not h.scheduler._loading
            assert h.calls["loading"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("task_name", ["_execute", "synthesize"])
def test_task_creation_failure_does_not_leave_a_pending_key(tmp_path, monkeypatch, task_name):
    async def scenario():
        async with harness(tmp_path) as h:
            original = asyncio.create_task

            def fail_selected(coroutine, *args, **kwargs):
                if coroutine.cr_code.co_name == task_name:
                    raise RuntimeError("Cannot start worker")
                return original(coroutine, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "create_task", fail_selected)
                with pytest.raises(SynthesisJobSubmissionError, match="Cannot start worker"):
                    await h.jobs.start(SynthesisRequest.model_validate(PAYLOAD), "one")
            await drain(h)
            replay = await submit(h)
            assert replay.status_code == 500
            assert replay.json() == {"detail": "Cannot start worker"}
            assert not h.jobs._admissions
            assert not any(h.calls.values())
            assert h.resources.reserved_cpu_units == 0

    asyncio.run(scenario())


def test_supervisor_cancellation_waits_for_native_worker_and_records_result(tmp_path):
    async def scenario():
        async with harness(tmp_path) as h:
            h.gates["generating"].clear()
            response = await submit(h)
            await wait_for_stage(h, "generating")
            supervisor = next(iter(h.jobs._tasks))
            supervisor.cancel()
            await asyncio.sleep(0)
            assert not supervisor.done()
            assert h.resources.reserved_cpu_units == 1
            h.gates["generating"].set()
            await drain(h)
            assert h.jobs.get(response.json()["jobId"]).status == "completed"
            assert h.resources.reserved_cpu_units == 0

    asyncio.run(scenario())
