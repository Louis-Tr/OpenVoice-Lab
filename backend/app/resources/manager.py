"""Atomic reservation ledger used by the processing scheduler."""

from dataclasses import dataclass
from threading import Lock

from app.resources.probe import ResourceProbe, ResourceSnapshot
from app.resources.profiles import ResourceDemand


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    fits: bool
    reason: str | None = None


@dataclass(slots=True)
class _Reservation:
    demand: ResourceDemand
    outstanding_memory_mb: float


class ResourceManager:
    """Combine live pressure with explicit CPU and memory commitments."""

    def __init__(
        self,
        probe: ResourceProbe,
        *,
        memory_headroom_mb: float | None,
        maximum_active_jobs: int,
        cpu_pressure_percent: float = 92.0,
    ) -> None:
        if maximum_active_jobs < 1:
            raise ValueError("maximum_active_jobs must be at least one")
        if memory_headroom_mb is not None and memory_headroom_mb < 0:
            raise ValueError("memory_headroom_mb cannot be negative")
        self._probe = probe
        initial = probe.sample()
        self._headroom_mb = memory_headroom_mb or max(
            512.0,
            initial.memory_limit_mb * 0.2,
        )
        self._maximum_active = maximum_active_jobs
        self._cpu_pressure_percent = cpu_pressure_percent
        self._reservations: dict[str, _Reservation] = {}
        self._snapshot = initial
        self._guard = Lock()

    @property
    def snapshot(self) -> ResourceSnapshot:
        return self._snapshot

    def refresh(self) -> ResourceSnapshot:
        snapshot = self._probe.sample()
        with self._guard:
            observed_growth = max(0.0, snapshot.memory_used_mb - self._snapshot.memory_used_mb)
            for reservation in self._reservations.values():
                materialized = min(observed_growth, reservation.outstanding_memory_mb)
                reservation.outstanding_memory_mb -= materialized
                observed_growth -= materialized
                if observed_growth <= 0:
                    break
            self._snapshot = snapshot
        return snapshot

    def can_ever_fit(self, demand: ResourceDemand) -> AdmissionDecision:
        snapshot = self._snapshot
        usable_memory = max(0.0, snapshot.memory_limit_mb - self._headroom_mb)
        if demand.memory_mb > usable_memory:
            return AdmissionDecision(False, "memory")
        if demand.cpu_cores > snapshot.cpu_limit_cores:
            return AdmissionDecision(False, "cpu")
        return AdmissionDecision(True)

    def can_admit(self, demand: ResourceDemand) -> AdmissionDecision:
        with self._guard:
            snapshot = self._snapshot
            reservations = tuple(self._reservations.values())
            if not snapshot.valid:
                return AdmissionDecision(False, "resource-measurement")
            if len(reservations) >= self._maximum_active:
                return AdmissionDecision(False, "worker-capacity")
            if demand.exclusive and reservations:
                return AdmissionDecision(False, "exclusive-work")
            if any(item.demand.exclusive for item in reservations):
                return AdmissionDecision(False, "exclusive-work")
            if any(item.demand.resource_key == demand.resource_key for item in reservations):
                return AdmissionDecision(False, "engine-busy")
            if snapshot.cpu_utilization_percent >= self._cpu_pressure_percent:
                return AdmissionDecision(False, "cpu")
            reserved_cpu = sum(item.demand.cpu_cores for item in reservations)
            if reserved_cpu + demand.cpu_cores > snapshot.cpu_limit_cores:
                return AdmissionDecision(False, "cpu")
            reserved_memory = sum(item.outstanding_memory_mb for item in reservations)
            ceiling = snapshot.memory_limit_mb - self._headroom_mb
            if snapshot.memory_used_mb + reserved_memory + demand.memory_mb > ceiling:
                return AdmissionDecision(False, "memory")
            return AdmissionDecision(True)

    def reserve(self, job_id: str, demand: ResourceDemand) -> None:
        with self._guard:
            if job_id in self._reservations:
                raise RuntimeError(f"Resources are already reserved for '{job_id}'.")
            self._reservations[job_id] = _Reservation(demand, demand.memory_mb)

    def release(self, job_id: str) -> None:
        with self._guard:
            self._reservations.pop(job_id, None)

    def status(self) -> dict[str, object]:
        with self._guard:
            snapshot = self._snapshot
            reservations = tuple(self._reservations.values())
        return {
            "source": snapshot.source,
            "valid": snapshot.valid,
            "memoryLimitMb": round(snapshot.memory_limit_mb, 1),
            "memoryUsedMb": round(snapshot.memory_used_mb, 1),
            "memoryHeadroomMb": round(self._headroom_mb, 1),
            "cpuLimitCores": round(snapshot.cpu_limit_cores, 2),
            "cpuUsedCores": round(snapshot.cpu_used_cores, 2),
            "activeJobs": len(reservations),
            "reservedMemoryMb": round(
                sum(item.outstanding_memory_mb for item in reservations), 1
            ),
            "reservedCpuCores": round(
                sum(item.demand.cpu_cores for item in reservations), 2
            ),
        }
