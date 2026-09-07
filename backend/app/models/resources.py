"""Conservative process-local CPU and RSS admission estimates (MiB)."""

import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import psutil

MIB = 1024**2
MEMORY_PROFILES = {"kokoro-fp32": 1700.0, "kokoro-fp16": 1500.0, "speecht5-pretrained": 1600.0}
INT8_UNAVAILABLE = "Kokoro INT8 admission is disabled until maximum-input memory is measured."


class ResourceUnavailableError(RuntimeError):
    """A request cannot currently reserve its engine or resource allowance."""


class CgroupLimits:
    """Read this process's v1/v2 cgroup and ancestor limits, when running on Linux."""

    def __init__(self, proc_root: Path = Path("/proc/self")) -> None:
        self._paths: dict[str, list[Path]] = {}
        if sys.platform != "linux":
            return
        try:
            memberships = [
                line.split(":", 2) for line in (proc_root / "cgroup").read_text().splitlines()
            ]
            for line in (proc_root / "mountinfo").read_text().splitlines():
                left, right = line.split(" - ", 1)
                fields, filesystem = left.split(), right.split()
                if filesystem[0] not in {"cgroup", "cgroup2"}:
                    continue
                root = Path(fields[3].replace("\\040", " "))
                mount = Path(fields[4].replace("\\040", " "))
                for _, controllers, member in memberships:
                    names = controllers.split(",") if controllers else ["v2"]
                    for name in names:
                        if name not in {"v2", "cpu", "memory"}:
                            continue
                        if name == "v2" and filesystem[0] != "cgroup2":
                            continue
                        if name != "v2" and name not in filesystem[2].split(","):
                            continue
                        member_path = Path(member)
                        if member_path.is_relative_to(root):
                            directory = mount / member_path.relative_to(root)
                        elif member == "/":  # cgroup namespace rooted at the mount
                            directory = mount
                        else:
                            continue
                        self._paths[name] = [
                            directory,
                            *[p for p in directory.parents if p.is_relative_to(mount)],
                        ]
        except (OSError, ValueError, IndexError):
            self._paths = {}

    def memory_headroom(self) -> float:
        readings = [math.inf]
        for kind, paths in self._paths.items():
            if kind not in {"v2", "memory"}:
                continue
            limit_file, usage_file = (
                ("memory.max", "memory.current")
                if kind == "v2"
                else ("memory.limit_in_bytes", "memory.usage_in_bytes")
            )
            for path in paths:
                try:
                    limit = int((path / limit_file).read_text())
                    usage = int((path / usage_file).read_text())
                    readings.append(max(0, limit - usage) / MIB)
                except (OSError, ValueError):
                    continue  # missing controller or unlimited ("max")
        return min(readings)

    def cpu_capacity(self) -> float:
        readings = [math.inf]
        for kind, paths in self._paths.items():
            if kind not in {"v2", "cpu"}:
                continue
            for path in paths:
                try:
                    if kind == "v2":
                        quota, period = (path / "cpu.max").read_text().split()
                    else:
                        quota = (path / "cpu.cfs_quota_us").read_text()
                        period = (path / "cpu.cfs_period_us").read_text()
                    if int(quota) > 0 and int(period) > 0:
                        readings.append(int(quota) / int(period))
                except (OSError, ValueError):
                    continue
        return min(readings)


class ResourceManager:
    """Reservations are mutated only under the owning scheduler's lock."""

    def __init__(
        self,
        *,
        profiles: Mapping[str, float] = MEMORY_PROFILES,
        cpu_units: int | None = None,
        cpu_threads: int | None = None,
        memory_limit_mb: float | None = None,
        memory_headroom_mb: float = 256,
        memory_reader: Callable[[], float] | None = None,
        rss_reader: Callable[[], float] | None = None,
        cgroups: CgroupLimits | None = None,
    ) -> None:
        self._cgroups = cgroups or CgroupLimits()
        process = psutil.Process()
        try:
            detected = len(process.cpu_affinity())
        except (AttributeError, psutil.Error):
            detected = psutil.cpu_count() or 1
        capacity = min(detected, self._cgroups.cpu_capacity())
        # Even fractional quotas need one runnable thread; the OS throttles it.
        detected_units = max(1, math.floor(capacity))
        self.cpu_units = detected_units if cpu_units is None else cpu_units
        self.cpu_threads = min(2, self.cpu_units) if cpu_threads is None else cpu_threads
        if not 1 <= self.cpu_units <= detected_units:
            raise ValueError("product_cpu_units must be positive and within usable CPU capacity")
        if not 1 <= self.cpu_threads <= self.cpu_units:
            raise ValueError("product_cpu_threads must be positive and <= product_cpu_units")
        if not math.isfinite(memory_headroom_mb) or memory_headroom_mb < 0:
            raise ValueError("product_memory_headroom_mb must be finite and non-negative")
        if memory_limit_mb is not None and (
            not math.isfinite(memory_limit_mb) or memory_limit_mb <= 0
        ):
            raise ValueError("product_memory_limit_mb must be finite and positive")
        self.profiles = dict(profiles)
        if any(not math.isfinite(value) or value <= 0 for value in self.profiles.values()):
            raise ValueError("Memory profiles must be finite and positive")
        self._memory_reader = memory_reader or (lambda: psutil.virtual_memory().available / MIB)
        self._rss_reader = rss_reader or (lambda: process.memory_info().rss / MIB)
        self._memory_limit_mb = memory_limit_mb
        self._headroom_mb = memory_headroom_mb
        self.reserved_memory_mb = 0.0
        self.reserved_cpu_units = 0

    def available_memory(self) -> float:
        headroom = min(self._memory_reader(), self._cgroups.memory_headroom())
        if self._memory_limit_mb is not None:
            headroom = min(headroom, self._memory_limit_mb - self._rss_reader())
        return max(0.0, headroom - self._headroom_mb - self.reserved_memory_mb)

    def estimate_required_memory(self, model_id: str) -> float:
        try:
            return self.profiles[model_id]
        except KeyError as error:
            raise ResourceUnavailableError(
                f"No memory admission profile for '{model_id}'."
            ) from error

    def available_cpu_units(self) -> int:
        return self.cpu_units - self.reserved_cpu_units

    def can_admit(self, model_id: str) -> bool:
        return (
            self.available_cpu_units() >= self.cpu_threads
            and self.available_memory() >= self.estimate_required_memory(model_id)
        )

    def reserve(self, model_id: str) -> None:
        self.reserved_memory_mb += self.estimate_required_memory(model_id)
        self.reserved_cpu_units += self.cpu_threads

    def release(self, model_id: str) -> None:
        self.reserved_memory_mb -= self.estimate_required_memory(model_id)
        self.reserved_cpu_units -= self.cpu_threads
