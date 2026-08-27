"""Read verified Stage 11 artifacts into a stable public report."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.experiments.common import ExperimentEvidenceError, read_json, sha256_file
from app.schemas.experiment import (
    ExperimentDatasetStats,
    ExperimentEvaluation,
    ExperimentIncident,
    ExperimentIntegrity,
    ExperimentLossPoint,
    ExperimentReport,
    ExperimentTrainingConfig,
    ExperimentVariantReport,
)

VARIANT_NAMES = {
    "v1-baseline": "V1 Baseline",
    "v2-term-balance": "V2 Term Balance",
    "v3-replay": "V3 Replay",
}


class ExperimentReportService:
    """Validate and project the immutable Stage 11 result evidence."""

    def __init__(self, artifact_root: Path, manifest_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()
        self._manifest_root = manifest_root.resolve()
        self._cached: ExperimentReport | None = None

    def get(self) -> ExperimentReport:
        if self._cached is None:
            self._cached = self._build()
        return self._cached.model_copy(deep=True)

    def _build(self) -> ExperimentReport:
        audit = read_json(self._artifact_root / "final-audit.json")
        lock = read_json(self._manifest_root / "dataset-lock.json")
        self._validate_audit(audit)
        if lock.get("status") != "passed":
            raise ExperimentEvidenceError("The Stage 11 dataset lock did not pass.")
        source_audit = lock.get("source_audit")
        split_counts = source_audit.get("split_counts") if isinstance(source_audit, dict) else None
        if not isinstance(split_counts, dict) or set(split_counts) != {
            "train",
            "validation",
            "test",
        }:
            raise ExperimentEvidenceError("The Stage 11 source split counts are incomplete.")

        variants: list[ExperimentVariantReport] = []
        shared_training: dict[str, Any] | None = None
        shared_models: dict[str, Any] | None = None
        configuration_sha256: str | None = None
        dataset_lock_sha256: str | None = None

        for variant_id, name in VARIANT_NAMES.items():
            audit_variant = audit["variants"].get(variant_id)
            lock_variant = lock["variants"].get(variant_id)
            if not isinstance(audit_variant, dict) or not isinstance(lock_variant, dict):
                raise ExperimentEvidenceError(f"Missing evidence for {variant_id}.")
            root = self._artifact_root / variant_id
            provenance = read_json(root / "run_provenance.json")
            training_metadata = read_json(root / "training_metadata.json")
            evaluation = read_json(root / "evaluation" / "summary.json")
            artifact_manifest_path = root / "run_artifact_manifest.json"
            artifact_manifest = read_json(artifact_manifest_path)

            self._validate_variant_identity(variant_id, audit_variant, provenance)
            shared_training = self._same_or_set(
                shared_training, provenance["training"], "training configuration"
            )
            shared_models = self._same_or_set(shared_models, provenance["models"], "model revisions")
            configuration_sha256 = self._same_scalar(
                configuration_sha256,
                provenance["configuration_sha256"],
                "configuration hash",
            )
            dataset_lock_sha256 = self._same_scalar(
                dataset_lock_sha256,
                provenance["dataset_lock_sha256"],
                "dataset-lock hash",
            )
            if provenance.get("dataset_lock_status") != "passed":
                raise ExperimentEvidenceError(f"Dataset lock failed for {variant_id}.")
            if provenance.get("stability_gate_decision") != audit["stability_gate_decision"]:
                raise ExperimentEvidenceError(f"Stability override differs for {variant_id}.")
            expected_manifest_sha = audit_variant["final_artifacts"]["manifest_sha256"]
            if sha256_file(artifact_manifest_path) != expected_manifest_sha:
                raise ExperimentEvidenceError(
                    f"Final artifact manifest hash differs for {variant_id}."
                )

            model_sha = self._selected_model_sha(artifact_manifest, root)
            loss_history = [
                ExperimentLossPoint(
                    step=int(item["step"]),
                    epoch=float(item["epoch"]),
                    evaluation_loss=float(item["eval_loss"]),
                )
                for item in training_metadata["log_history"]
                if "eval_loss" in item
            ]
            distribution = lock_variant["distribution"]
            summary_accuracy = evaluation["domain_term_accuracy"]
            synthesis = evaluation["synthesis_verification"]
            variants.append(
                ExperimentVariantReport(
                    id=variant_id,
                    name=name,
                    pod_id=audit_variant["pod_id"],
                    started_at=audit_variant["pod_created_utc"],
                    finished_at=audit_variant["pod_terminated_utc"],
                    final_step=int(audit_variant["global_step"]),
                    best_step=int(audit_variant["best_step"]),
                    best_validation_loss=float(audit_variant["best_eval_loss"]),
                    stopped_early=bool(audit_variant["stopped_early"]),
                    training_seconds=float(training_metadata["train_metrics"]["train_runtime"]),
                    training_steps_per_second=float(
                        training_metadata["train_metrics"]["train_steps_per_second"]
                    ),
                    checkpoint_steps=[int(item["step"]) for item in audit_variant["checkpoints"]],
                    estimated_cost_usd=float(audit_variant["estimated_cost_usd"]),
                    dataset=ExperimentDatasetStats(
                        strategy=lock_variant["strategy"],
                        scheduled_rows=int(distribution["scheduled_rows"]),
                        unique_source_rows=int(distribution["unique_source_rows"]),
                        repeated_exposures=int(distribution["repeated_exposures"]),
                        rows_with_terms=int(distribution["rows_with_terms"]),
                        duration_hours=float(distribution["duration_hours"]),
                        unique_speakers=int(distribution["unique_speakers"]),
                        source_pool_counts=distribution["source_pool_counts"],
                    ),
                    validation_history=loss_history,
                    evaluation=ExperimentEvaluation(
                        case_count=int(evaluation["case_count"]),
                        failure_count=int(evaluation["failure_count"]),
                        domain_terms_correct=int(summary_accuracy["correct"]),
                        domain_terms_total=int(summary_accuracy["total"]),
                        domain_term_accuracy=float(summary_accuracy["accuracy"]),
                        word_error_rate=float(evaluation["wer"]),
                        average_inference_ms=float(evaluation["average_inference_ms"]),
                        average_real_time_factor=float(evaluation["average_rtf"]),
                        peak_gpu_memory_mb=float(evaluation["peak_gpu_memory_mb"]),
                        synthesis_verified=synthesis.get("status") == "passed",
                    ),
                    selected_model_sha256=model_sha,
                    artifact_manifest_sha256=audit_variant["final_artifacts"]["manifest_sha256"],
                )
            )

        assert shared_training is not None
        assert shared_models is not None
        assert configuration_sha256 is not None
        assert dataset_lock_sha256 is not None
        if dataset_lock_sha256 != sha256_file(self._manifest_root / "dataset-lock.json"):
            raise ExperimentEvidenceError("Dataset-lock hash differs between run evidence and lock file.")
        self._validate_shared_test_manifest(lock)
        stopping = read_json(self._artifact_root / "v1-baseline" / "run_provenance.json")[
            "early_stopping"
        ]
        incidents = [ExperimentIncident.model_validate(item) for item in audit["controller_incidents"]]
        return ExperimentReport(
            run_id=audit["run_id"],
            experiment_id="stage11-speecht5-full",
            headline=(
                "V3 Replay delivered the strongest domain-term accuracy and lowest WER, "
                "with a measurable runtime tradeoff."
            ),
            runtime_label="Historical secure RTX 4090 evaluation",
            integrity=ExperimentIntegrity(
                status="passed",
                dataset_lock_verified=True,
                checkpoint_hashes_verified=bool(audit["all_checkpoint_hashes_verified"]),
                final_artifact_hashes_verified=bool(audit["all_final_artifact_hashes_verified"]),
                all_pods_terminated=bool(audit["all_pods_terminated"]),
                checkpoint_count=int(audit["checkpoint_count"]),
                stability_gate_decision=audit["stability_gate_decision"],
            ),
            training=ExperimentTrainingConfig(
                precision="BF16" if shared_training["bf16"] else "FP32",
                physical_batch_size=int(shared_training["physical_batch_size"]),
                gradient_accumulation_steps=int(
                    shared_training["gradient_accumulation_steps"]
                ),
                effective_batch_size=int(shared_training["effective_batch_size"]),
                maximum_steps=int(shared_training["max_steps"]),
                nominal_epochs=int(shared_training["nominal_epochs"]),
                learning_rate=float(shared_training["learning_rate"]),
                warmup_steps=int(shared_training["warmup_steps"]),
                evaluation_steps=int(shared_training["evaluation_steps"]),
                maximum_gradient_norm=float(shared_training["max_grad_norm"]),
                gradient_checkpointing=bool(shared_training["gradient_checkpointing"]),
                early_stopping_patience=int(stopping["patience_evaluations"]),
                early_stopping_threshold=float(stopping["threshold"]),
                seed=int(lock["seed"]),
            ),
            shared_splits={
                key: int(value) for key, value in split_counts.items()
            },
            dataset_lock_sha256=dataset_lock_sha256,
            configuration_sha256=configuration_sha256,
            source_model_revisions={
                "tts": shared_models["tts_revision"],
                "vocoder": shared_models["vocoder_revision"],
                "speakerEncoder": shared_models["speaker_encoder_revision"],
                "asr": shared_models["asr_revision"],
            },
            variants=variants,
            incidents=incidents,
            training_resumptions=audit["training_resumptions"],
            total_gpu_hours=float(audit["total_gpu_hours"]),
            estimated_total_cost_usd=float(audit["estimated_total_cost_usd"]),
        )

    @staticmethod
    def _validate_audit(audit: dict[str, Any]) -> None:
        required_true = (
            "all_checkpoint_hashes_verified",
            "all_final_artifact_hashes_verified",
            "all_pods_terminated",
        )
        if audit.get("status") != "passed" or any(not audit.get(field) for field in required_true):
            raise ExperimentEvidenceError("The Stage 11 final audit is not fully verified.")
        if int(audit.get("checkpoint_count", 0)) != 24:
            raise ExperimentEvidenceError("Stage 11 does not contain all 24 checkpoints.")

    @staticmethod
    def _validate_variant_identity(
        variant_id: str, audit: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        if audit.get("status") != "completed" or not audit.get("pod_terminated"):
            raise ExperimentEvidenceError(f"Variant {variant_id} is not complete and terminated.")
        if provenance.get("variant") != variant_id:
            raise ExperimentEvidenceError(f"Variant provenance mismatch for {variant_id}.")

    @staticmethod
    def _same_or_set(
        current: dict[str, Any] | None, incoming: dict[str, Any], label: str
    ) -> dict[str, Any]:
        if current is not None and current != incoming:
            raise ExperimentEvidenceError(f"Stage 11 variants disagree on {label}.")
        return deepcopy(incoming)

    @staticmethod
    def _same_scalar(current: str | None, incoming: str, label: str) -> str:
        if current is not None and current != incoming:
            raise ExperimentEvidenceError(f"Stage 11 variants disagree on {label}.")
        return incoming

    @staticmethod
    def _selected_model_sha(manifest: dict[str, Any], root: Path) -> str:
        entry = next(
            (
                item
                for item in manifest.get("files", [])
                if item.get("path") == "selected-model/model.safetensors"
            ),
            None,
        )
        model = root / "selected-model" / "model.safetensors"
        if (
            entry is None
            or not model.is_file()
            or model.stat().st_size != int(entry["bytes"])
            or sha256_file(model) != entry["sha256"]
        ):
            raise ExperimentEvidenceError(f"Selected model evidence is incomplete: {model}")
        return str(entry["sha256"])

    @staticmethod
    def _validate_shared_test_manifest(lock: dict[str, Any]) -> None:
        values = {
            item["manifest_sha256"]["test"] for item in lock["variants"].values()
        }
        if len(values) != 1 or not lock.get("shared_evaluation_manifests"):
            raise ExperimentEvidenceError("The Stage 11 test manifest is not shared.")
