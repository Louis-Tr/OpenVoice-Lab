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
V1C_CONFIG_PATH = (
    REPOSITORY_ROOT / "training" / "config" / "v1c_gradual_unfreeze.yaml"
)


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


def test_gradual_unfreeze_recipe_is_locked_to_v1_data_and_twenty_five_step_saves() -> None:
    config = load_config(V1C_CONFIG_PATH)
    report = build_preflight_report(config)
    values = config.values

    assert report["configuration_valid"] is True
    assert report["launch_ready"] is True
    assert list(report["manifests"]) == ["v1c-gradual-unfreeze"]
    assert values["variants"][0]["dataset_source_variant"] == "v1-baseline"
    assert values["approach"]["transition_after_step"] == 50
    assert values["approach"]["head_learning_rate"] == 1e-6
    assert values["approach"]["decoder_learning_rate"] == 5e-7
    assert values["training"]["max_steps"] == 250
    assert values["training"]["evaluation_steps"] == 25
    assert values["checkpoints"]["recovery_interval_steps"] == 25
    assert values["checkpoints"]["durable_milestone_steps"] == list(
        range(25, 251, 25)
    )
    assert report["manifests"]["v1c-gradual-unfreeze"]["train"]["sha256"] == (
        "6e610a77bbef8b462700cf5b08214b00c6eea5218dffbb42e1f87d92d281880e"
    )
