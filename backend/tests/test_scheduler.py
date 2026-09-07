"""Deterministic lease, concurrency, capacity and cleanup coverage."""

import asyncio
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Barrier, Event

import pytest
from test_model_loader import CloseableEngine, definition

from app.audio.service import AudioService
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader, ModelLoadError
from app.models.registry import ModelRegistry
from app.models.resources import ResourceManager, ResourceUnavailableError
from app.models.scheduler import EngineScheduler
from app.schemas.synthesis import SynthesisRequest
from app.synthesis.service import SynthesisService
from app.text_processing.service import TextProcessingService


def make_scheduler(
    factory=lambda model: CloseableEngine(model.key), *, slots=None, units=2, memory=lambda: 10_000
):
    resources = ResourceManager(
        profiles={"a": 100, "b": 100, "c": 100},
        cpu_units=units,
        cpu_threads=1,
        memory_headroom_mb=0,
        memory_reader=memory,
    )
    loader = ModelLoader(engine_factory=factory)
    return EngineScheduler(loader, resources, MetricsCollector(), maximum_cached_engines=slots)


@pytest.mark.parametrize("same_model", [True, False])
def test_parallel_inference_reuses_weights(tmp_path, same_model):
    barrier = Barrier(2)

    class ConcurrentEngine(CloseableEngine):
        def synthesize(self, *args, **kwargs):
            barrier.wait(timeout=5)
            return super().synthesize(*args, **kwargs)

    scheduler = make_scheduler(lambda m: ConcurrentEngine(m.key))
    first = definition("a", tmp_path / "a")
    second = first if same_model else definition("b", tmp_path / "b")
    for model in {first, second}:
        with scheduler.acquire(model):
            pass

    def run(model):
        with scheduler.acquire(model) as lease:
            engine = lease.loaded.value
            result = engine.synthesize("hello", "voice")
            return engine, result

    with ThreadPoolExecutor(2) as pool:
        one, two = pool.submit(run, first), pool.submit(run, second)
        a, b = one.result(10), two.result(10)
    assert (a[0] is b[0]) == same_model
    assert a[1].samples is not b[1].samples
    assert scheduler._loader.load_count("a") == 1
    assert scheduler.resources.reserved_cpu_units == 0
    assert scheduler.resources.reserved_memory_mb == 0


def test_active_engine_not_evicted_and_idle_evicted_before_construction(tmp_path):
    engines = {}

    def construct(model):
        if model.key == "b":
            assert engines["a"].closed
        engines[model.key] = CloseableEngine(model.key)
        return engines[model.key]

    scheduler = make_scheduler(construct, slots=1)
    a, b = definition("a", tmp_path / "a"), definition("b", tmp_path / "b")
    with scheduler.acquire(a):
        with pytest.raises(ResourceUnavailableError), scheduler.acquire(b):
            pass
        assert not engines["a"].closed
    with scheduler.acquire(b):
        assert engines["a"].closed


def test_pending_load_is_reserved_without_holding_scheduler_lock(tmp_path):
    entered, finish = Event(), Event()

    def construct(model):
        entered.set()
        assert finish.wait(5)
        return CloseableEngine(model.key)

    scheduler = make_scheduler(construct, slots=1)
    a, b = definition("a", tmp_path / "a"), definition("b", tmp_path / "b")

    def run():
        with scheduler.acquire(a):
            pass

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(run)
        try:
            assert entered.wait(5)
            assert scheduler.resources.reserved_memory_mb == 100
            for model in (a, b):
                with pytest.raises(ResourceUnavailableError), scheduler.acquire(model):
                    pass
        finally:
            finish.set()
        pending.result(5)


@pytest.mark.parametrize("units,memory", [(1, 10_000), (2, 150)])
def test_atomic_warm_admission(tmp_path, units, memory):
    scheduler = make_scheduler(units=units, memory=lambda: memory)
    model = definition("a", tmp_path / "a")
    with (
        scheduler.acquire(model),
        pytest.raises(ResourceUnavailableError),
        scheduler.acquire(model),
    ):
        pass
    assert scheduler.resources.reserved_cpu_units == 0


def test_simultaneous_admissions_cannot_overbook(tmp_path):
    scheduler = make_scheduler(units=1)
    model = definition("a", tmp_path / "a")
    with scheduler.acquire(model):
        pass
    start, finish = Barrier(3), Event()

    def run():
        start.wait(5)
        try:
            with scheduler.acquire(model):
                assert finish.wait(5)
                return True
        except ResourceUnavailableError:
            return False

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        try:
            start.wait(5)
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            assert len(done) == 1
            assert next(iter(done)).result() is False
            assert scheduler.resources.reserved_cpu_units == 1
            assert scheduler.resources.reserved_memory_mb == 100
        finally:
            finish.set()
        assert sorted(future.result(5) for future in futures) == [False, True]
    assert scheduler.resources.reserved_cpu_units == 0


def test_eviction_rechecks_memory_and_does_not_assume_freed_bytes(tmp_path):
    available = [1000]
    scheduler = make_scheduler(memory=lambda: available[0])
    a, b = definition("a", tmp_path / "a"), definition("b", tmp_path / "b")
    with scheduler.acquire(a) as lease:
        engine = lease.loaded.value
    available[0] = 50
    with pytest.raises(ResourceUnavailableError), scheduler.acquire(b):
        pass
    assert engine.closed
    assert scheduler._loader.load_count("b") == 0


def test_failure_releases_reservations_and_retry_works(tmp_path):
    fail = [True]

    def construct(model):
        if fail[0]:
            raise ValueError("load failure")
        return CloseableEngine(model.key)

    scheduler = make_scheduler(construct)
    model = definition("a", tmp_path / "a")
    with pytest.raises(ModelLoadError), scheduler.acquire(model):
        pass
    assert not scheduler._loading
    assert scheduler.resources.reserved_cpu_units == 0
    fail[0] = False
    with pytest.raises(RuntimeError), scheduler.acquire(model):
        raise RuntimeError("inference failure")
    with scheduler.acquire(model) as lease:
        assert lease.warm
    assert scheduler.resources.reserved_memory_mb == 0


def test_int8_and_unknown_profiles_rejected(tmp_path):
    scheduler = make_scheduler()
    for identifier in ("kokoro-q8", "unknown"):
        with (
            pytest.raises(ResourceUnavailableError, match="profile"),
            scheduler.acquire(definition(identifier, tmp_path / identifier)),
        ):
            pass


def make_service(tmp_path, scheduler, audio):
    model = definition("a", tmp_path / "a")
    return SynthesisService(
        ModelRegistry([model]), scheduler, audio, MetricsCollector(), TextProcessingService()
    )


def test_audio_failure_keeps_lease_until_storage_returns(tmp_path):
    scheduler = make_scheduler()

    class FailedAudio(AudioService):
        def create_artifact(self, *args, **kwargs):
            assert scheduler._records["a"].active_users == 1
            raise RuntimeError("storage failure")

    service = make_service(tmp_path, scheduler, FailedAudio(tmp_path / "audio"))
    with pytest.raises(RuntimeError, match="storage failure"):
        service._synthesize_sync(SynthesisRequest(text="hi", model_id="a", voice_id="voice"))
    assert scheduler._records["a"].active_users == 0
    assert scheduler.resources.reserved_cpu_units == 0


def test_http_task_cancellation_does_not_release_running_thread(tmp_path):
    entered, finish, released = Event(), Event(), Event()

    class BlockingEngine(CloseableEngine):
        def synthesize(self, *args, **kwargs):
            entered.set()
            assert finish.wait(5)
            return super().synthesize(*args, **kwargs)

    scheduler = make_scheduler(lambda model: BlockingEngine(model.key))
    original_acquire = scheduler.acquire

    @contextmanager
    def observed_acquire(model):
        try:
            with original_acquire(model) as lease:
                yield lease
        finally:
            released.set()

    scheduler.acquire = observed_acquire
    service = make_service(tmp_path, scheduler, AudioService(tmp_path / "audio"))

    async def scenario():
        task = asyncio.create_task(
            service.synthesize(SynthesisRequest(text="hi", model_id="a", voice_id="voice"))
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert scheduler._records["a"].active_users == 1
            assert scheduler.resources.reserved_cpu_units == 1
        finally:
            finish.set()
        assert await asyncio.to_thread(released.wait, 5)
        assert scheduler.resources.reserved_cpu_units == 0

    asyncio.run(scenario())
