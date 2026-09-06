"""Container-aware CPU and memory measurements."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """One normalized view of the resources available to this process."""

    memory_limit_mb: float
    memory_used_mb: float
    cpu_limit_cores: float
    cpu_used_cores: float
    captured_at: float
    source: str
    valid: bool = True

    @property
    def memory_available_mb(self) -> float:
        return max(0.0, self.memory_limit_mb - self.memory_used_mb)

    @property
    def cpu_utilization_percent(self) -> float:
        if self.cpu_limit_cores <= 0:
            return 100.0
        return max(0.0, self.cpu_used_cores / self.cpu_limit_cores * 100.0)


class ResourceProbe:
    """Read cgroup limits in containers and process-tree usage elsewhere."""

    _UNLIMITED_V1 = 1 << 60

    def __init__(
        self,
        *,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        memory_limit_mb: float | None = None,
        cpu_limit_cores: float | None = None,
        clock=time.monotonic,
    ) -> None:
        if memory_limit_mb is not None and memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if cpu_limit_cores is not None and cpu_limit_cores <= 0:
            raise ValueError("cpu_limit_cores must be positive")
        self._root = cgroup_root
        self._memory_override = memory_limit_mb
        self._cpu_override = cpu_limit_cores
        self._clock = clock
        self._last_cpu_seconds: float | None = None
        self._last_cpu_at: float | None = None

    def sample(self) -> ResourceSnapshot:
        """Return a nonblocking sample, falling back conservatively on read errors."""
        now = self._clock()
        try:
            cgroup = self._read_cgroup()
            if cgroup is not None:
                memory_limit, memory_used, cpu_limit, source = cgroup
            else:
                memory = psutil.virtual_memory()
                memory_limit = memory.total / (1024 * 1024)
                memory_used = self._process_tree_memory_mb()
                cpu_limit = float(psutil.cpu_count(logical=True) or 1)
                source = "host/process-tree"

            if self._memory_override is not None:
                memory_limit = min(memory_limit, self._memory_override)
                source += "+configured-memory"
            if self._cpu_override is not None:
                cpu_limit = min(cpu_limit, self._cpu_override)
                source += "+configured-cpu"

            cpu_used = self._sample_process_cpu(now)
            return ResourceSnapshot(
                memory_limit_mb=max(1.0, memory_limit),
                memory_used_mb=max(0.0, memory_used),
                cpu_limit_cores=max(0.1, cpu_limit),
                cpu_used_cores=max(0.0, cpu_used),
                captured_at=now,
                source=source,
            )
        except (OSError, ValueError, psutil.Error):
            memory_limit = self._memory_override or 512.0
            cpu_limit = self._cpu_override or 1.0
            return ResourceSnapshot(
                memory_limit_mb=memory_limit,
                memory_used_mb=memory_limit,
                cpu_limit_cores=cpu_limit,
                cpu_used_cores=cpu_limit,
                captured_at=now,
                source="conservative-fallback",
                valid=False,
            )

    def _read_cgroup(self) -> tuple[float, float, float, str] | None:
        if (self._root / "cgroup.controllers").exists():
            memory_limit_raw = self._read_text(self._root / "memory.max")
            memory_current_raw = self._read_text(self._root / "memory.current")
            cpu_max = self._read_text(self._root / "cpu.max")
            if memory_limit_raw and memory_current_raw and cpu_max:
                host_memory = psutil.virtual_memory().total
                memory_limit_bytes = (
                    host_memory if memory_limit_raw == "max" else int(memory_limit_raw)
                )
                quota, period = cpu_max.split()
                quota_cores = (
                    float(psutil.cpu_count(logical=True) or 1)
                    if quota == "max"
                    else int(quota) / int(period)
                )
                cpuset_cores = self._cpuset_count(
                    self._read_text(self._root / "cpuset.cpus.effective")
                    or self._read_text(self._root / "cpuset.cpus")
                )
                return (
                    memory_limit_bytes / (1024 * 1024),
                    int(memory_current_raw) / (1024 * 1024),
                    min(quota_cores, cpuset_cores or quota_cores),
                    "cgroup-v2",
                )

        memory_limit_path = self._root / "memory" / "memory.limit_in_bytes"
        memory_used_path = self._root / "memory" / "memory.usage_in_bytes"
        quota_path = self._root / "cpu" / "cpu.cfs_quota_us"
        period_path = self._root / "cpu" / "cpu.cfs_period_us"
        if memory_limit_path.exists() and memory_used_path.exists():
            memory_limit_bytes = int(self._read_text(memory_limit_path) or 0)
            if memory_limit_bytes <= 0 or memory_limit_bytes >= self._UNLIMITED_V1:
                memory_limit_bytes = psutil.virtual_memory().total
            quota = int(self._read_text(quota_path) or -1)
            period = int(self._read_text(period_path) or 100_000)
            cpu_limit = (
                quota / period if quota > 0 else float(psutil.cpu_count(logical=True) or 1)
            )
            return (
                memory_limit_bytes / (1024 * 1024),
                int(self._read_text(memory_used_path) or 0) / (1024 * 1024),
                cpu_limit,
                "cgroup-v1",
            )
        return None

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None

    @staticmethod
    def _cpuset_count(value: str | None) -> int | None:
        if not value:
            return None
        count = 0
        for section in value.split(","):
            bounds = section.split("-", maxsplit=1)
            count += int(bounds[-1]) - int(bounds[0]) + 1
        return count or None

    @staticmethod
    def _process_tree_memory_mb() -> float:
        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        total = 0
        for child in processes:
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / (1024 * 1024)

    def _sample_process_cpu(self, now: float) -> float:
        process = psutil.Process()
        total = 0.0
        for child in [process, *process.children(recursive=True)]:
            try:
                times = child.cpu_times()
                total += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        previous_total = self._last_cpu_seconds
        previous_at = self._last_cpu_at
        self._last_cpu_seconds = total
        self._last_cpu_at = now
        if previous_total is None or previous_at is None or now <= previous_at:
            return 0.0
        return max(0.0, (total - previous_total) / (now - previous_at))
