from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FullTrainingConfig:
    repository_root: Path
    config_path: Path
    values: dict[str, Any]

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.repository_root / path).resolve()

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError(f"Could not find repository root above {config_path}")


def load_config(path: str | Path) -> FullTrainingConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("full training config must be a schema_version: 1 mapping")
    config = FullTrainingConfig(_repository_root(config_path), config_path, values)
    errors = validate_values(values)
    if errors:
        raise ValueError("invalid full training config: " + "; ".join(errors))
    return config


def validate_values(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    training = values.get("training", {})
    checkpoints = values.get("checkpoints", {})
    early_stopping = values.get("early_stopping", {})
    dataset = values.get("dataset", {})
    runtime = values.get("runtime", {})

    physical = int(training.get("physical_batch_size", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    gpu_count = int(runtime.get("gpu_count", 0))
    effective = int(training.get("effective_batch_size", 0))
    expected_effective = physical * accumulation * gpu_count
    if expected_effective <= 0 or effective != expected_effective:
        errors.append(
            f"effective_batch_size must equal physical x accumulation x GPUs "
            f"({expected_effective}), got {effective}"
        )

    block_size = int(dataset.get("schedule_block_size", 0))
    if block_size <= 0 or physical % block_size != 0 or effective % block_size != 0:
        errors.append(
            "physical and effective batch sizes must be divisible by schedule_block_size"
        )
    if dataset.get("sampler") != "deterministic_block_shuffle":
        errors.append(
            "the locked variant schedules require deterministic_block_shuffle"
        )

    max_steps = int(training.get("max_steps", 0))
    warmup_steps = int(training.get("warmup_steps", -1))
    evaluation_steps = int(training.get("evaluation_steps", 0))
    recovery_steps = int(checkpoints.get("recovery_interval_steps", 0))
    if max_steps <= 0:
        errors.append("max_steps must be positive")
    if not 0 <= warmup_steps < max_steps:
        errors.append("warmup_steps must be non-negative and lower than max_steps")
    if evaluation_steps <= 0 or recovery_steps <= 0:
        errors.append("evaluation and recovery checkpoint intervals must be positive")
    elif evaluation_steps != recovery_steps:
        errors.append(
            "evaluation_steps must equal recovery_interval_steps for best-model loading"
        )

    milestones = [int(step) for step in checkpoints.get("durable_milestone_steps", [])]
    if milestones != sorted(set(milestones)):
        errors.append("durable checkpoint milestones must be unique and sorted")
    if not milestones or milestones[-1] != max_steps:
        errors.append("the final durable checkpoint milestone must equal max_steps")
    if any(step <= 0 or step % recovery_steps != 0 for step in milestones):
        errors.append("every durable milestone must align with a recovery checkpoint")

    if bool(training.get("bf16")) == bool(training.get("fp16")):
        errors.append("exactly one of bf16 and fp16 must be enabled")
    if not bool(training.get("gradient_checkpointing")):
        errors.append("gradient_checkpointing must remain enabled for the 4090 run")
    if bool(training.get("model_use_cache")):
        errors.append("model_use_cache must be false with gradient checkpointing")
    if early_stopping.get("metric") != "eval_loss":
        errors.append("early stopping and best-model selection must use eval_loss")
    if int(early_stopping.get("patience_evaluations", 0)) <= 0:
        errors.append("early stopping patience must be positive")

    reduction_factor = int(training.get("reduction_factor", 2))
    if reduction_factor <= 0:
        errors.append("training.reduction_factor must be positive")

    approach = values.get("approach")
    if approach is not None and approach.get("type") == "gradual_unfreeze":
        transition = int(approach.get("transition_after_step", 0))
        if transition <= 0 or transition >= max_steps:
            errors.append("gradual-unfreeze transition must occur within training")
        head_lr = float(approach.get("head_learning_rate", 0))
        decoder_lr = float(approach.get("decoder_learning_rate", 0))
        if head_lr <= 0 or decoder_lr <= 0 or decoder_lr >= head_lr:
            errors.append(
                "gradual-unfreeze decoder LR must be positive and lower than head LR"
            )
        if int(approach.get("top_decoder_blocks", 0)) != 2:
            errors.append("gradual-unfreeze must expose exactly two top decoder blocks")
        if not approach.get("phase_one_trainable_prefixes") or not approach.get(
            "phase_two_decoder_prefixes"
        ):
            errors.append("gradual-unfreeze trainable prefixes are required")
    elif approach is not None and approach.get("type") == "peft_lora":
        if int(approach.get("rank", 0)) <= 0:
            errors.append("LoRA rank must be positive")
        if int(approach.get("alpha", 0)) <= 0:
            errors.append("LoRA alpha must be positive")
        if not approach.get("target_modules"):
            errors.append("LoRA target_modules are required")
        if int(approach.get("expected_target_linear_modules", 0)) <= 0:
            errors.append("LoRA expected_target_linear_modules must be positive")
    elif approach is not None:
        errors.append("the configured V1 approach type is unsupported")

    variants = values.get("variants", [])
    variant_ids = [
        variant.get("id") for variant in variants if isinstance(variant, dict)
    ]
    if not variant_ids or any(not value for value in variant_ids):
        errors.append("at least one named training variant is required")
    elif len(variant_ids) != len(set(variant_ids)):
        errors.append("training variant ids must be unique")
    if values.get("experiment_id") == "stage11-speecht5-full" and variant_ids != [
        "v1-baseline",
        "v2-term-balance",
        "v3-replay",
    ]:
        errors.append(
            "the original full experiment must contain v1-baseline, "
            "v2-term-balance, and v3-replay in order"
        )
    if len({variant.get("output_root") for variant in variants}) != len(variants):
        errors.append("each variant must use an independent output_root")

    gate = values.get("stability_gate", {})
    if int(gate.get("required_physical_batch_size", 0)) != physical:
        errors.append(
            "stability gate batch size must match the training physical batch size"
        )
    if int(gate.get("required_gradient_accumulation_steps", 0)) != accumulation:
        errors.append("stability gate accumulation must match training")
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_preflight_report(config: FullTrainingConfig) -> dict[str, Any]:
    values = config.values
    errors: list[str] = []
    warnings: list[str] = []
    dataset = values["dataset"]
    training = values["training"]

    lock_path = config.resolve(dataset["lock"])
    lock: dict[str, Any] = {}
    if not lock_path.is_file():
        errors.append(f"dataset lock is missing: {lock_path}")
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "passed":
            errors.append("dataset lock status is not passed")

    manifests: dict[str, Any] = {}
    expected_rows = dataset["expected_rows"]
    locked_variants = lock.get("variants", {})
    for variant in values["variants"]:
        variant_id = variant["id"]
        dataset_lock_variant = variant.get(
            "dataset_source_variant",
            variant.get(
                "dataset_lock_variant", variant.get("dataset_variant_id", variant_id)
            ),
        )
        root = config.resolve(variant["manifest_root"])
        manifests[variant_id] = {}
        for split in ("train", "validation", "test"):
            path = root / f"{split}.jsonl"
            entry: dict[str, Any] = {"path": str(path)}
            if not path.is_file():
                errors.append(f"missing {variant_id} {split} manifest: {path}")
                entry["status"] = "missing"
            else:
                row_count = _jsonl_rows(path)
                digest = _sha256(path)
                expected_digest = (
                    locked_variants.get(dataset_lock_variant, {})
                    .get("manifest_sha256", {})
                    .get(split)
                )
                entry.update(rows=row_count, sha256=digest, status="passed")
                if row_count != int(expected_rows[split]):
                    errors.append(
                        f"{variant_id} {split} has {row_count} rows; "
                        f"expected {expected_rows[split]}"
                    )
                    entry["status"] = "failed"
                if digest != expected_digest:
                    errors.append(
                        f"{variant_id} {split} digest does not match dataset lock "
                        f"entry {dataset_lock_variant}"
                    )
                    entry["status"] = "failed"
            manifests[variant_id][split] = entry

    train_rows = int(expected_rows["train"])
    effective_batch = int(training["effective_batch_size"])
    optimizer_steps_per_epoch = math.ceil(train_rows / effective_batch)
    max_steps = int(training["max_steps"])

    stability_path = config.resolve(values["stability_gate"]["result"])
    stability: dict[str, Any] = {"path": str(stability_path), "matching": False}
    if not stability_path.is_file():
        warnings.append("no stability probe result exists for the final physical batch")
        stability["status"] = "missing"
    else:
        result = json.loads(stability_path.read_text(encoding="utf-8"))
        observed = result.get("configuration", {})
        required = values["stability_gate"]
        matching = (
            result.get("status") == "passed"
            and int(observed.get("batch_size", 0))
            == int(required["required_physical_batch_size"])
            and int(observed.get("gradient_accumulation_steps", 0))
            == int(required["required_gradient_accumulation_steps"])
            and bool(observed.get(required["required_precision"]))
        )
        stability.update(
            status=result.get("status"),
            observed_physical_batch_size=observed.get("batch_size"),
            observed_gradient_accumulation_steps=observed.get(
                "gradient_accumulation_steps"
            ),
            matching=matching,
        )
        if not matching and bool(values["stability_gate"]["required"]):
            warnings.append(
                "existing stability result passed at physical batch 4, not the final batch 16; "
                "run the final stability gate before provisioning full training"
            )
        elif not matching:
            warnings.append(
                "batch-16 stability evidence is absent; the recorded user-authorized "
                "override permits launch"
            )

    configuration_valid = not errors
    stability_required = bool(values["stability_gate"]["required"])
    return {
        "schema_version": 1,
        "experiment_id": values["experiment_id"],
        "configuration_sha256": config.digest,
        "configuration_valid": configuration_valid,
        "launch_ready": configuration_valid
        and (stability["matching"] or not stability_required),
        "errors": errors,
        "warnings": warnings,
        "derived": {
            "physical_batch_size": training["physical_batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "effective_batch_size": effective_batch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "approximate_epochs_at_max_steps": round(
                max_steps / optimizer_steps_per_epoch, 6
            ),
            "maximum_optimizer_steps": max_steps,
            "evaluation_count_at_max_steps": max_steps
            // int(training["evaluation_steps"]),
            "durable_checkpoint_steps": values["checkpoints"][
                "durable_milestone_steps"
            ],
        },
        "dataset_lock": {
            "path": str(lock_path),
            "status": lock.get("status", "missing"),
            "sha256": _sha256(lock_path) if lock_path.is_file() else None,
        },
        "manifests": manifests,
        "stability_gate": stability,
    }
