"""Memory bounds, cgroup discovery and conservative resource accounting."""

import math
from types import SimpleNamespace

import pytest

from app.models import resources as module
from app.models.resources import CgroupLimits, ResourceManager


@pytest.fixture(autouse=True)
def cpu_capacity(monkeypatch):
    monkeypatch.setattr(
        module.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: list(range(8)))
    )


def manager(**kwargs):
    return ResourceManager(
        profiles={"model": 100}, memory_reader=lambda: 1000, rss_reader=lambda: 200, **kwargs
    )


def test_memory_uses_tightest_limit_and_subtracts_full_reservations():
    cgroups = SimpleNamespace(memory_headroom=lambda: 700, cpu_capacity=lambda: 4)
    resource = manager(cgroups=cgroups, memory_limit_mb=800, memory_headroom_mb=50, cpu_threads=2)
    assert resource.available_memory() == 550
    assert resource.cpu_units == 4
    resource.reserve("model")
    assert resource.available_memory() == 450
    assert resource.available_cpu_units() == 2
    assert resource.can_admit("model")
    resource.reserve("model")
    assert not resource.can_admit("model")
    resource.release("model")
    resource.release("model")
    assert resource.available_memory() == 550


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu_units": 0},
        {"cpu_units": 9},
        {"cpu_threads": 0},
        {"cpu_units": 2, "cpu_threads": 3},
        {"memory_limit_mb": -1},
        {"memory_headroom_mb": -1},
        {"memory_headroom_mb": math.nan},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        manager(**kwargs)


def test_fractional_cpu_quota_requires_one_throttled_thread():
    resource = manager(
        cgroups=SimpleNamespace(memory_headroom=lambda: math.inf, cpu_capacity=lambda: 0.5)
    )
    assert resource.cpu_units == resource.cpu_threads == 1


@pytest.mark.parametrize("v2", [True, False])
def test_cgroup_membership_and_ancestor_limits(tmp_path, monkeypatch, v2):
    monkeypatch.setattr(module.sys, "platform", "linux")
    proc = tmp_path / "proc"
    proc.mkdir()
    mount = tmp_path / "cgroup"
    child = mount / "child"
    child.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/child\n" if v2 else "2:cpu,memory:/child\n")
    filesystem = "cgroup2 cgroup rw" if v2 else "cgroup cgroup rw,cpu,memory"
    (proc / "mountinfo").write_text(f"1 0 0:1 / {mount.as_posix()} rw - {filesystem}\n")
    if v2:
        (mount / "memory.max").write_text(str(800 * module.MIB))
        (mount / "memory.current").write_text(str(300 * module.MIB))
        (child / "memory.max").write_text("max")
        (mount / "cpu.max").write_text("250000 100000")
        (child / "cpu.max").write_text("max 100000")
    else:
        (mount / "memory.limit_in_bytes").write_text(str(800 * module.MIB))
        (mount / "memory.usage_in_bytes").write_text(str(300 * module.MIB))
        (mount / "cpu.cfs_quota_us").write_text("250000")
        (mount / "cpu.cfs_period_us").write_text("100000")
    groups = CgroupLimits(proc)
    assert groups.memory_headroom() == 500
    assert groups.cpu_capacity() == 2.5


def test_no_cgroups_has_no_extra_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    groups = CgroupLimits(tmp_path)
    assert groups.memory_headroom() == math.inf
    assert groups.cpu_capacity() == math.inf
