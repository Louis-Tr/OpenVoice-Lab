"""Shared engine leases with atomic resource admission and idle-only eviction."""

import gc
import json
import logging
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

from app.inference.base import TTSInferenceEngine
from app.metrics.collector import MeasuredModelLoad, MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition
from app.models.resources import ResourceManager, ResourceUnavailableError

LOGGER = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class EngineRecord:
    engine: TTSInferenceEngine
    active_users: int
    memory_profile: float


@dataclass(frozen=True, slots=True)
class EngineLease:
    loaded: MeasuredModelLoad[TTSInferenceEngine]
    warm: bool


class EngineScheduler:
    def __init__(
        self,
        loader: ModelLoader,
        resources: ResourceManager,
        metrics: MetricsCollector,
        *,
        maximum_cached_engines: int | None = None,
    ) -> None:
        if maximum_cached_engines is not None and maximum_cached_engines < 1:
            raise ValueError("maximum_cached_engines must be positive")
        self._loader, self.resources, self._metrics = loader, resources, metrics
        self._maximum = maximum_cached_engines
        self._records: OrderedDict[str, EngineRecord] = OrderedDict()
        self._loading: set[str] = set()
        self._retiring: set[str] = set()
        self._lock = Lock()

    def _log(self, event: str, model_id: str, **extra: object) -> None:
        record = self._records.get(model_id)
        fields = {
            "event": event,
            "model_id": model_id,
            "active_users": record.active_users if record else 0,
            "reserved_cpu_units": self.resources.reserved_cpu_units,
            "reserved_memory_mb": self.resources.reserved_memory_mb,
            "memory_estimate_mb": self.resources.profiles.get(model_id),
            **extra,
        }
        LOGGER.info(json.dumps(fields), extra=fields)

    def _reject(self, model_id: str, reason: str) -> None:
        self._log("engine_rejected", model_id, reason=reason)
        raise ResourceUnavailableError(reason)

    def _reserve(self, model: ModelDefinition) -> EngineRecord | None:
        while True:
            with self._lock:
                model_id = model.key
                if model_id not in self.resources.profiles:
                    self._reject(model_id, f"No memory admission profile for '{model_id}'.")
                if not model.artifacts_available:
                    self._reject(
                        model_id, model.resolved_unavailable_reason or "Model unavailable."
                    )
                if model_id in self._loading or model_id in self._retiring:
                    self._reject(model_id, "Model is loading or being evicted; retry later.")
                if self.resources.available_cpu_units() < self.resources.cpu_threads:
                    self._reject(model_id, "Insufficient CPU capacity; retry later.")
                cached = self._records.get(model_id)
                slots = len(self._records) + len(self._loading) + len(self._retiring)
                slot_available = (
                    cached is not None or self._maximum is None or slots < self._maximum
                )
                if slot_available and self.resources.can_admit(model_id):
                    self.resources.reserve(model_id)
                    if cached is None:
                        self._loading.add(model_id)
                    else:
                        cached.active_users += 1
                        self._records.move_to_end(model_id)
                    self._log("engine_admitted", model_id, warm=cached is not None)
                    return cached
                victim_id = next(
                    (
                        key
                        for key, record in self._records.items()
                        if record.active_users == 0 and key != model_id
                    ),
                    None,
                )
                if victim_id is None:
                    self._reject(
                        model_id, "Insufficient memory or no idle cache slot; retry later."
                    )
                victim = self._records.pop(victim_id)
                self._retiring.add(victim_id)
            # Cleanup can be slow: it must not hold the admission lock.
            try:
                try:
                    self._dispose(victim)
                finally:
                    del victim
                    gc.collect()
            finally:
                with self._lock:
                    self._retiring.remove(victim_id)
                    self._log("engine_evicted", victim_id)
            # Recheck actual headroom and competing reservations after cleanup.

    @staticmethod
    def _dispose(record: EngineRecord) -> None:
        close = getattr(record.engine, "close", None)
        if callable(close):
            close()

    @contextmanager
    def acquire(self, model: ModelDefinition) -> Iterator[EngineLease]:
        record = self._reserve(model)
        warm = record is not None
        try:
            if record is None:
                loaded = self._metrics.measure_model_load(lambda: self._loader.load(model))
                record = EngineRecord(
                    loaded.value, 1, self.resources.estimate_required_memory(model.key)
                )
                with self._lock:
                    self._loading.remove(model.key)
                    self._records[model.key] = record
            else:
                loaded = self._metrics.measure_model_load(lambda: record.engine)
            yield EngineLease(loaded, warm)
        finally:
            with self._lock:
                self._loading.discard(model.key)
                if record is not None:
                    record.active_users -= 1
                self.resources.release(model.key)
                self._log("engine_released", model.key)
