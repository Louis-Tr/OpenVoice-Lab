"""Module-level coverage for probing, admission, backfilling, and lifecycle."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from app.resources.manager import ResourceManager
from app.resources.probe import ResourceProbe, ResourceSnapshot
from app.resources.profiles import ResourceDemand
from app.scheduling.policy import select_next
from app.scheduling.service import (
    ProcessingJobCancelledError,
    ProcessingQueueFullError,
    ProcessingScheduler,
    ScheduledWork,
)


class StaticProbe:
    def __init__(self, snapshots: list[ResourceSnapshot]) -> None:
        self.snapshots = snapshots
        self.index = 0

    def sample(self) -> ResourceSnapshot:
        snapshot = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return snapshot


def snapshot(memory_used: float = 100.0) -> ResourceSnapshot:
    return ResourceSnapshot(1000.0, memory_used, 4.0, 0.0, 1.0, "test")


@dataclass
class Candidate:
    id: str
    enqueued_at: float
    sequence: int
    fits: bool


def test_resource_modules_read_cgroups_and_reconcile_reserved_growth(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("cpu memory", encoding="utf-8")
    (tmp_path / "memory.max").write_text(str(2 * 1024 * 1024 * 1024), encoding="utf-8")
    (tmp_path / "memory.current").write_text(str(512 * 1024 * 1024), encoding="utf-8")
    (tmp_path / "cpu.max").write_text("150000 100000", encoding="utf-8")
    (tmp_path / "cpuset.cpus.effective").write_text("0-3", encoding="utf-8")
    measured = ResourceProbe(cgroup_root=tmp_path).sample()
    assert measured.source == "cgroup-v2"
    assert measured.memory_limit_mb == 2048
    assert measured.memory_used_mb == 512
    assert measured.cpu_limit_cores == 1.5

    probe = StaticProbe([snapshot(100), snapshot(100), snapshot(250)])
    manager = ResourceManager(probe, memory_headroom_mb=100, maximum_active_jobs=3)
    manager.refresh()
    demand = ResourceDemand(300, 1, "model-a")
    manager.reserve("one", demand)
    assert manager.status()["reservedMemoryMb"] == 300
    manager.refresh()
    assert manager.status()["reservedMemoryMb"] == 150
    assert manager.can_admit(ResourceDemand(400, 1, "model-b")).fits


def test_policy_backfills_then_reserves_the_oldest_waiter_at_thirty_seconds() -> None:
    oldest = Candidate("oldest", 0.0, 1, False)
    later = Candidate("later", 10.0, 2, True)
    before = select_next(
        [oldest, later],
        now=29.999,
        aging_threshold_seconds=30.0,
        fits=lambda item: item.fits,
    )
    assert before.selected_id == "later"
    assert before.reserved_id is None

    at_threshold = select_next(
        [oldest, later],
        now=30.0,
        aging_threshold_seconds=30.0,
        fits=lambda item: item.fits,
    )
    assert at_threshold.selected_id is None
    assert at_threshold.reserved_id == "oldest"


def test_scheduler_module_backfills_bounds_and_cancels_without_blocking_threads() -> None:
    async def scenario() -> None:
        manager = ResourceManager(
            StaticProbe([snapshot()] * 30),
            memory_headroom_mb=100,
            maximum_active_jobs=2,
        )
        scheduler = ProcessingScheduler(
            manager,
            aging_threshold_seconds=30,
            queue_capacity=1,
            maximum_workers=2,
            sample_interval_seconds=0.01,
        )
        release = Event()
        small_ran = Event()
        blocker = await scheduler.submit(
            ScheduledWork(
                "blocker",
                ResourceDemand(400, 1, "blocker"),
                lambda: release.wait(2),
            )
        )
        for _ in range(50):
            if "blocker" in scheduler.status()["activeJobIds"]:
                break
            await asyncio.sleep(0.005)

        large = await scheduler.submit(
            ScheduledWork("large", ResourceDemand(600, 1, "large"), lambda: "large")
        )
        with pytest.raises(ProcessingQueueFullError):
            await scheduler.submit(
                ScheduledWork("overflow", ResourceDemand(100, 1, "overflow"), lambda: None)
            )
        assert await scheduler.cancel("large") is True
        with pytest.raises(ProcessingJobCancelledError):
            await large

        small = await scheduler.submit(
            ScheduledWork(
                "small",
                ResourceDemand(300, 1, "small"),
                lambda: small_ran.set() or "small",
            )
        )
        assert await asyncio.wait_for(small, 1) == "small"
        assert small_ran.is_set()
        assert not blocker.done()
        release.set()
        await asyncio.wait_for(blocker, 1)
        await scheduler.close()

    asyncio.run(scenario())
