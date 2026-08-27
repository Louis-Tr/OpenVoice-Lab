from __future__ import annotations

import copy
from pathlib import Path

import yaml

from training.full_training.config import (
    build_preflight_report,
    load_config,
    validate_values,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "training" / "config" / "full_training.yaml"


def _values() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_final_batch_math_and_checkpoint_quarters_are_locked() -> None:
    values = _values()

    assert validate_values(values) == []
    assert values["training"]["physical_batch_size"] == 16
    assert values["training"]["gradient_accumulation_steps"] == 2
    assert values["training"]["effective_batch_size"] == 32
    assert values["checkpoints"]["durable_milestone_steps"] == [250, 500, 750, 1000]


def test_invalid_effective_batch_is_rejected() -> None:
    values = copy.deepcopy(_values())
    values["training"]["effective_batch_size"] = 16

    assert any("effective_batch_size" in error for error in validate_values(values))


def test_milestones_must_align_with_recovery_checkpoints() -> None:
    values = copy.deepcopy(_values())
    values["checkpoints"]["durable_milestone_steps"] = [249, 500, 750, 1000]

    assert any("milestone" in error for error in validate_values(values))


def test_current_locked_manifests_pass_preflight() -> None:
    config = load_config(CONFIG_PATH)
    report = build_preflight_report(config)

    assert report["configuration_valid"] is True
    assert report["dataset_lock"]["status"] == "passed"
    assert report["derived"]["optimizer_steps_per_epoch"] == 166
    assert report["derived"]["approximate_epochs_at_max_steps"] == 6.024096
    assert not report["errors"]


def test_user_authorized_override_allows_launch_without_batch_sixteen_probe() -> None:
    config = load_config(CONFIG_PATH)
    report = build_preflight_report(config)

    assert report["stability_gate"]["observed_physical_batch_size"] == 4
    assert report["stability_gate"]["matching"] is False
    assert config.values["stability_gate"]["override"] == "user_authorized_skip"
    assert report["launch_ready"] is True
