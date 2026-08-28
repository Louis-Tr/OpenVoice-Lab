"""Project the four completed V1 approach runs into the experiment API."""

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.experiments.approach_catalog import V1_APPROACHES
from app.experiments.common import ExperimentEvidenceError, read_json, sha256_file
from app.schemas.experiment import (
    ExperimentApproachTrainingConfig,
    ExperimentDataAudit,
    ExperimentDatasetStats,
    ExperimentEvaluation,
    ExperimentIncident,
    ExperimentIntegrity,
    ExperimentLossPoint,
    ExperimentPretrainedReport,
    ExperimentReport,
    ExperimentTrainingConfig,
    ExperimentVariantReport,
)


class V1ApproachReportService:
    """Validate and present the four pinned V1 approach runs."""

    def __init__(
        self,
        approach_run_root: Path,
        pretrained_artifact_root: Path,
        manifest_root: Path,
    ) -> None:
        self._run_root = approach_run_root.resolve()
        self._pretrained_root = pretrained_artifact_root.resolve()
        self._manifest_root = manifest_root.resolve()
        self._cached: ExperimentReport | None = None

    def get(self) -> ExperimentReport:
        if self._cached is None:
            self._cached = self._build()
        return self._cached.model_copy(deep=True)

    def _build(self) -> ExperimentReport:
        lock_path = self._manifest_root / "dataset-lock.json"
        lock = read_json(lock_path)
        if lock.get("status") != "passed":
            raise ExperimentEvidenceError("The Stage 11 dataset lock did not pass.")
        dataset_lock_sha256 = sha256_file(lock_path)
        source_audit = self._source_audit(lock)
        locked_dataset = lock.get("variants", {}).get("v1-baseline")
        if not isinstance(locked_dataset, dict):
            raise ExperimentEvidenceError("The locked V1 dataset is unavailable.")
        locked_hashes = locked_dataset.get("manifest_sha256")
        distribution = locked_dataset.get("distribution")
        if not isinstance(locked_hashes, dict) or not isinstance(distribution, dict):
            raise ExperimentEvidenceError("The locked V1 dataset evidence is incomplete.")

        variants: list[ExperimentVariantReport] = []
        incidents: list[ExperimentIncident] = []
        resumptions: list[dict[str, object]] = []
        config_hashes: list[str] = []
        shared_models: dict[str, Any] | None = None
        shared_training: dict[str, Any] | None = None
        total_gpu_hours = 0.0
        total_cost = 0.0

        for definition in V1_APPROACHES:
            run_root = self._run_root / definition.run_id
            final_root = run_root / "final"
            run = read_json(run_root / "run.json")
            preflight = read_json(run_root / "preflight.json")
            pod = read_json(run_root / "pod.json")
            status = read_json(run_root / "status.json")
            checkpoint_inventory = read_json(run_root / "checkpoint-inventory.json")
            final_download = read_json(run_root / "final-download.json")
            provenance = read_json(final_root / "run_provenance.json")
            metadata = read_json(final_root / "training_metadata.json")
            evaluation = read_json(final_root / "evaluation" / "summary.json")
            selection = read_json(final_root / "selected-model" / "selection.json")
            approach_runtime = read_json(final_root / "approach_runtime.json")
            marker = read_json(final_root / "RUN_COMPLETE.json")
            artifact_manifest_path = final_root / "run_artifact_manifest.json"
            artifact_manifest = read_json(artifact_manifest_path)

            self._validate_run(
                definition.variant_id,
                definition.run_id,
                run,
                preflight,
                pod,
                status,
                final_download,
                provenance,
                marker,
                artifact_manifest_path,
                dataset_lock_sha256,
                locked_hashes,
            )
            checkpoint_steps = self._checkpoint_steps(
                checkpoint_inventory, definition.run_id
            )
            model_sha = self._selected_model_sha(artifact_manifest, final_root)
            training = provenance.get("training")
            stopping = provenance.get("early_stopping")
            models = provenance.get("models")
            if not isinstance(training, dict) or not isinstance(stopping, dict):
                raise ExperimentEvidenceError(
                    f"Training configuration is incomplete for {definition.variant_id}."
                )
            if not isinstance(models, dict):
                raise ExperimentEvidenceError(
                    f"Model provenance is incomplete for {definition.variant_id}."
                )
            if shared_models is None:
                shared_models = models
            elif shared_models != models:
                raise ExperimentEvidenceError("The four approaches use different model revisions.")
            shared_training = self._merge_shared_training(shared_training, training)

            config_sha = str(run.get("profile", {}).get("config_sha256", ""))
            if len(config_sha) != 64:
                raise ExperimentEvidenceError(
                    f"Configuration hash is missing for {definition.variant_id}."
                )
            config_hashes.append(config_sha)
            best_step = self._checkpoint_step(metadata.get("best_model_checkpoint"))
            selected_step = int(selection["selected_step"])
            train_metrics = metadata.get("train_metrics")
            if not isinstance(train_metrics, dict):
                raise ExperimentEvidenceError(
                    f"Training metrics are incomplete for {definition.variant_id}."
                )
            accuracy = evaluation.get("domain_term_accuracy")
            synthesis = evaluation.get("synthesis_verification")
            if not isinstance(accuracy, dict) or not isinstance(synthesis, dict):
                raise ExperimentEvidenceError(
                    f"Evaluation metrics are incomplete for {definition.variant_id}."
                )
            if int(evaluation.get("case_count", -1)) != int(
                source_audit["split_counts"]["test"]
            ):
                raise ExperimentEvidenceError(
                    f"The shared test case count differs for {definition.variant_id}."
                )

            dataset = ExperimentDatasetStats(
                strategy=str(locked_dataset["strategy"]),
                scheduled_rows=int(distribution["scheduled_rows"]),
                unique_source_rows=int(distribution["unique_source_rows"]),
                repeated_exposures=int(distribution["repeated_exposures"]),
                rows_with_terms=int(distribution["rows_with_terms"]),
                duration_hours=float(distribution["duration_hours"]),
                unique_speakers=int(distribution["unique_speakers"]),
                maximum_speaker_share=float(distribution["maximum_speaker_share"]),
                source_pool_counts=distribution["source_pool_counts"],
                term_category_occurrences=distribution["term_category_occurrences"],
                manifest_sha256={key: str(value) for key, value in locked_hashes.items()},
            )
            approach_type = str(
                approach_runtime.get("type")
                or training.get("approach")
                or definition.variant_id
            )
            variants.append(
                ExperimentVariantReport(
                    id=definition.variant_id,
                    name=definition.name,
                    run_id=definition.run_id,
                    approach=approach_type,
                    pod_id=str(pod["pod_id"]),
                    started_at=pod["created_utc"],
                    finished_at=pod["terminated_utc"],
                    final_step=int(metadata["global_step"]),
                    best_step=best_step,
                    best_validation_loss=float(metadata["best_metric"]),
                    stopped_early=bool(metadata["stopped_early"]),
                    training_seconds=float(train_metrics["train_runtime"]),
                    training_steps_per_second=float(train_metrics["train_steps_per_second"]),
                    train_loss=float(train_metrics["train_loss"]),
                    checkpoint_steps=checkpoint_steps,
                    estimated_cost_usd=float(status["estimated_cost_usd"]),
                    dataset=dataset,
                    training_config=ExperimentApproachTrainingConfig(
                        type=approach_type,
                        physical_batch_size=int(training["physical_batch_size"]),
                        gradient_accumulation_steps=int(
                            training["gradient_accumulation_steps"]
                        ),
                        effective_batch_size=int(training["effective_batch_size"]),
                        learning_rate=float(training["learning_rate"]),
                        early_stopping_enabled=bool(stopping["enabled"]),
                        early_stopping_patience=int(stopping["patience_evaluations"]),
                        reduction_factor=int(
                            approach_runtime.get("configured_reduction_factor", 2)
                        ),
                    ),
                    validation_history=[
                        ExperimentLossPoint(
                            step=int(item["step"]),
                            epoch=float(item["epoch"]),
                            evaluation_loss=float(item["eval_loss"]),
                        )
                        for item in metadata["log_history"]
                        if "eval_loss" in item
                    ],
                    evaluation=ExperimentEvaluation(
                        case_count=int(evaluation["case_count"]),
                        failure_count=int(evaluation["failure_count"]),
                        domain_terms_correct=int(accuracy["correct"]),
                        domain_terms_total=int(accuracy["total"]),
                        domain_term_accuracy=float(accuracy["accuracy"]),
                        word_error_rate=float(evaluation["wer"]),
                        average_inference_ms=float(evaluation["average_inference_ms"]),
                        average_real_time_factor=float(evaluation["average_rtf"]),
                        peak_gpu_memory_mb=float(evaluation["peak_gpu_memory_mb"]),
                        peak_process_memory_mb=float(
                            evaluation["peak_process_memory_mb"]
                        ),
                        successful_cases=int(evaluation["successful_cases"]),
                        exact_sentence_rate=float(evaluation["exact_sentence_rate"]),
                        synthesis_verified=synthesis.get("status") == "passed",
                    ),
                    selected_model_sha256=model_sha,
                    artifact_manifest_sha256=sha256_file(artifact_manifest_path),
                    selected_step=selected_step,
                    selection_status=str(selection["selection_status"]),
                )
            )

            start = self._date(pod["created_utc"])
            finish = self._date(pod["terminated_utc"])
            total_gpu_hours += (finish - start).total_seconds() / 3600
            total_cost += float(status["estimated_cost_usd"])
            incidents.extend(self._compatibility_incidents(run_root, definition.variant_id))
            if metadata.get("resumed_from"):
                resumptions.append(
                    {
                        "variant": definition.variant_id,
                        "checkpoint": metadata["resumed_from"],
                    }
                )

        assert shared_models is not None
        assert shared_training is not None
        pretrained = self._pretrained_report(
            lock=lock,
            dataset_lock_sha256=dataset_lock_sha256,
            shared_models=shared_models,
        )
        combined_configuration_sha256 = sha256(
            "|".join(config_hashes).encode("utf-8")
        ).hexdigest()
        leakage = source_audit["leakage_intersections"]
        leakage_count = sum(
            int(value)
            for comparisons in leakage.values()
            for value in comparisons.values()
        )
        training_reference = variants[0].training_config
        assert training_reference is not None

        return ExperimentReport(
            run_id="stage11-v1-approaches-20260828",
            experiment_id="stage11-speecht5-v1-approaches",
            headline=(
                "Pretrained retained the quality lead: V1C tied domain-term accuracy, "
                "while V1B delivered the fastest inference with a measurable quality trade-off."
            ),
            runtime_label="Verified secure RTX 4090 shared-test evaluation",
            integrity=ExperimentIntegrity(
                status="passed",
                dataset_lock_verified=True,
                checkpoint_hashes_verified=True,
                final_artifact_hashes_verified=True,
                all_pods_terminated=True,
                checkpoint_count=sum(len(item.checkpoint_steps) for item in variants),
                stability_gate_decision="User-authorized approach-specific probe overrides",
            ),
            training=ExperimentTrainingConfig(
                precision="BF16" if shared_training["bf16"] else "FP32",
                physical_batch_size=None,
                gradient_accumulation_steps=None,
                effective_batch_size=int(shared_training["effective_batch_size"]),
                maximum_steps=int(shared_training["max_steps"]),
                nominal_epochs=int(shared_training["nominal_epochs"]),
                learning_rate=None,
                warmup_steps=int(shared_training["warmup_steps"]),
                evaluation_steps=int(shared_training["evaluation_steps"]),
                maximum_gradient_norm=float(shared_training["max_grad_norm"]),
                gradient_checkpointing=bool(shared_training["gradient_checkpointing"]),
                early_stopping_patience=None,
                early_stopping_threshold=0.001,
                seed=int(lock["seed"]),
            ),
            data_audit=ExperimentDataAudit(
                status="passed",
                unique_audio_files=int(source_audit["unique_audio_files"]),
                audio_verification_failure_count=len(
                    source_audit["audio_verification_failures"]
                ),
                leakage_intersection_count=leakage_count,
                leakage_identity_fields=sorted(leakage),
                shared_evaluation_manifests=bool(lock["shared_evaluation_manifests"]),
                schedule_block_size=int(lock["batch_size"]),
                source_manifest_sha256={
                    key: str(value)
                    for key, value in lock["source_manifest_sha256"].items()
                },
                builder_sha256=str(lock["builder_sha256"]),
                variant_config_sha256=str(lock["config_sha256"]),
            ),
            shared_splits={
                key: int(value) for key, value in source_audit["split_counts"].items()
            },
            dataset_lock_sha256=dataset_lock_sha256,
            configuration_sha256=combined_configuration_sha256,
            source_model_revisions={
                "tts": shared_models["tts_revision"],
                "vocoder": shared_models["vocoder_revision"],
                "speakerEncoder": shared_models["speaker_encoder_revision"],
                "asr": shared_models["asr_revision"],
            },
            pretrained_control=pretrained,
            variants=variants,
            incidents=incidents,
            training_resumptions=resumptions,
            total_gpu_hours=total_gpu_hours,
            estimated_total_cost_usd=total_cost,
        )

    @staticmethod
    def _source_audit(lock: dict[str, Any]) -> dict[str, Any]:
        audit = lock.get("source_audit")
        if not isinstance(audit, dict) or audit.get("status") != "passed":
            raise ExperimentEvidenceError("The Stage 11 source audit is incomplete.")
        required = (
            "split_counts",
            "leakage_intersections",
            "audio_verification_failures",
            "unique_audio_files",
        )
        if any(field not in audit for field in required):
            raise ExperimentEvidenceError("The Stage 11 source audit fields are incomplete.")
        return audit

    @staticmethod
    def _validate_run(
        variant_id: str,
        run_id: str,
        run: dict[str, Any],
        preflight: dict[str, Any],
        pod: dict[str, Any],
        status: dict[str, Any],
        final_download: dict[str, Any],
        provenance: dict[str, Any],
        marker: dict[str, Any],
        artifact_manifest_path: Path,
        dataset_lock_sha256: str,
        locked_hashes: dict[str, Any],
    ) -> None:
        if (
            run.get("run_id") != run_id
            or run.get("approach") != variant_id
            or not run.get("artifacts_verified")
            or not run.get("pod_terminated")
            or run.get("status") != "terminated"
        ):
            raise ExperimentEvidenceError(f"Run document is incomplete for {variant_id}.")
        if (
            preflight.get("launch_ready") is not True
            or preflight.get("dataset_lock", {}).get("status") != "passed"
            or preflight.get("dataset_lock", {}).get("sha256") != dataset_lock_sha256
        ):
            raise ExperimentEvidenceError(f"Preflight evidence differs for {variant_id}.")
        if (
            pod.get("status") != "TERMINATED"
            or status.get("provider_status") != "TERMINATED"
            or pod.get("pod_id") != provenance.get("pod_id")
        ):
            raise ExperimentEvidenceError(f"Pod termination is unverified for {variant_id}.")
        if not final_download.get("verified") or final_download.get("manifest_errors"):
            raise ExperimentEvidenceError(f"Final download is unverified for {variant_id}.")
        if (
            provenance.get("variant") != variant_id
            or provenance.get("run_id") != run_id
            or provenance.get("dataset_lock_status") != "passed"
            or provenance.get("dataset_lock_sha256") != dataset_lock_sha256
            or provenance.get("manifest_sha256") != locked_hashes
        ):
            raise ExperimentEvidenceError(f"Run provenance differs for {variant_id}.")
        manifest_sha = sha256_file(artifact_manifest_path)
        if (
            marker.get("status") not in {"completed", "completed_with_failures"}
            or marker.get("variant") != variant_id
            or marker.get("artifact_manifest_sha256") != manifest_sha
        ):
            raise ExperimentEvidenceError(f"Completion marker differs for {variant_id}.")

    @staticmethod
    def _checkpoint_steps(inventory: dict[str, Any], run_id: str) -> list[int]:
        if inventory.get("run_id") != run_id or not isinstance(
            inventory.get("checkpoints"), dict
        ):
            raise ExperimentEvidenceError(f"Checkpoint inventory is invalid for {run_id}.")
        checkpoints = inventory["checkpoints"]
        if len(checkpoints) != 10 or any(
            not item.get("verified") or not Path(item.get("path", "")).is_dir()
            for item in checkpoints.values()
        ):
            raise ExperimentEvidenceError(f"Checkpoint verification failed for {run_id}.")
        return sorted(int(name.removeprefix("checkpoint-")) for name in checkpoints)

    @staticmethod
    def _selected_model_sha(manifest: dict[str, Any], final_root: Path) -> str:
        entry = next(
            (
                item
                for item in manifest.get("files", [])
                if item.get("path") == "selected-model/model.safetensors"
            ),
            None,
        )
        model = final_root / "selected-model" / "model.safetensors"
        if (
            entry is None
            or not model.is_file()
            or model.stat().st_size != int(entry.get("bytes", -1))
            or sha256_file(model) != entry.get("sha256")
        ):
            raise ExperimentEvidenceError(f"Selected model evidence is incomplete: {model}")
        return str(entry["sha256"])

    @staticmethod
    def _checkpoint_step(value: object) -> int:
        match = re.search(r"checkpoint-(\d+)", str(value))
        if match is None:
            raise ExperimentEvidenceError("The best validation checkpoint is missing.")
        return int(match.group(1))

    @staticmethod
    def _merge_shared_training(
        current: dict[str, Any] | None, incoming: dict[str, Any]
    ) -> dict[str, Any]:
        shared_fields = (
            "bf16",
            "effective_batch_size",
            "max_steps",
            "nominal_epochs",
            "warmup_steps",
            "evaluation_steps",
            "max_grad_norm",
            "gradient_checkpointing",
        )
        projected = {field: incoming.get(field) for field in shared_fields}
        if current is not None and current != projected:
            raise ExperimentEvidenceError(
                "The four approaches disagree on a declared shared hyperparameter."
            )
        return projected

    @staticmethod
    def _date(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _compatibility_incidents(
        run_root: Path, variant_id: str
    ) -> list[ExperimentIncident]:
        events_path = run_root / "events.jsonl"
        failures = 0
        passed = False
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                failures += event.get("event") == "compatibility_failed"
                passed = passed or event.get("event") == "compatibility_passed"
        if failures == 0:
            return []
        if not passed:
            raise ExperimentEvidenceError(
                f"The compatibility gate never passed for {variant_id}."
            )
        return [
            ExperimentIncident(
                id=f"{variant_id}-compatibility-gate",
                variants=[variant_id],
                impact=(
                    f"{failures} pre-training compatibility probe attempt"
                    f"{'s' if failures != 1 else ''} failed; model training had not started."
                ),
                resolution=(
                    "The reusable compatibility probe was repaired and passed before the "
                    "training process launched."
                ),
                training_restart=False,
            )
        ]

    def _pretrained_report(
        self,
        *,
        lock: dict[str, Any],
        dataset_lock_sha256: str,
        shared_models: dict[str, Any],
    ) -> ExperimentPretrainedReport:
        root = self._pretrained_root / "pretrained"
        marker = read_json(root / "EVALUATION_COMPLETE.json")
        provenance = read_json(root / "run_provenance.json")
        summary = read_json(root / "evaluation" / "summary.json")
        manifest_path = root / "evaluation_artifact_manifest.json"
        manifest_sha = sha256_file(manifest_path)
        if (
            marker.get("status") not in {"completed", "completed_with_failures"}
            or marker.get("evaluation_artifact_manifest_sha256") != manifest_sha
        ):
            raise ExperimentEvidenceError("The pretrained evaluation is incomplete.")
        test_hashes = {
            item["manifest_sha256"]["test"] for item in lock["variants"].values()
        }
        if len(test_hashes) != 1:
            raise ExperimentEvidenceError("The Stage 11 test manifest is not shared.")
        test_sha = next(iter(test_hashes))
        if (
            provenance.get("role") != "pretrained-control"
            or provenance.get("dataset_lock_sha256") != dataset_lock_sha256
            or provenance.get("test_manifest_sha256") != test_sha
            or provenance.get("models") != shared_models
        ):
            raise ExperimentEvidenceError("The pretrained provenance differs.")
        expected_cases = int(lock["source_audit"]["split_counts"]["test"])
        if int(summary.get("case_count", -1)) != expected_cases:
            raise ExperimentEvidenceError("The pretrained case count differs.")
        accuracy = summary["domain_term_accuracy"]
        synthesis = summary["synthesis_verification"]
        return ExperimentPretrainedReport(
            name="SpeechT5 Pretrained",
            model_id=str(shared_models["tts_id"]),
            revision=str(shared_models["tts_revision"]),
            evaluated_at=provenance["completed_utc"],
            pod_id=str(provenance["pod_id"]),
            hardware=str(provenance["environment"]["gpu"]),
            test_manifest_sha256=str(test_sha),
            artifact_manifest_sha256=manifest_sha,
            evaluation=ExperimentEvaluation(
                case_count=int(summary["case_count"]),
                failure_count=int(summary["failure_count"]),
                domain_terms_correct=int(accuracy["correct"]),
                domain_terms_total=int(accuracy["total"]),
                domain_term_accuracy=float(accuracy["accuracy"]),
                word_error_rate=float(summary["wer"]),
                average_inference_ms=float(summary["average_inference_ms"]),
                average_real_time_factor=float(summary["average_rtf"]),
                peak_gpu_memory_mb=float(summary["peak_gpu_memory_mb"]),
                successful_cases=int(summary["successful_cases"]),
                synthesis_verified=synthesis.get("status") == "passed",
            ),
        )
