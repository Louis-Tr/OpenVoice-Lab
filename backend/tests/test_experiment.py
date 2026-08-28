"""Stage 12 artifact, scoring, job, and API contract tests."""

import asyncio
import hashlib
import json
import wave
from pathlib import Path
from threading import Event

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.api.experiments import create_router
from app.experiments.artifact_report import ExperimentReportService
from app.experiments.common import ExperimentEvidenceError
from app.experiments.fixtures import ExperimentFixtureService
from app.experiments.jobs import ExperimentJobService
from app.experiments.model_registry import (
    ExperimentModelDefinition,
    ExperimentModelRegistry,
)
from app.experiments.scorer import score_terms, word_error_rate
from app.experiments.service import ExperimentService
from app.experiments.store import ExperimentJobStore
from app.experiments.v1_approach_report import V1ApproachReportService
from app.inference.speecht5_cpu import SpeechT5SynthesisOutput
from app.schemas.experiment import ExperimentComparisonRequest
from app.text_processing.service import TextProcessingService

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "stage11" / "full-training"
MANIFEST_ROOT = REPO_ROOT / "data-processing" / "manifests" / "stage11"
APPROACH_RUN_ROOT = REPO_ROOT / "artifacts" / "stage11" / "agent-runs"
VARIANTS = ("v1-baseline", "v2-term-balance", "v3-replay")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def experiment_evidence(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny, hash-consistent Stage 11 evidence tree for portable tests."""
    artifact_root = tmp_path / "stage11-artifacts"
    manifest_root = tmp_path / "stage11-manifests"
    fixture = {
        "sample_id": "medical-fixture-1",
        "text": "The patient has arm pain.",
        "medical_terms": [{"text": "arm", "canonical": "arm", "category": "anatomy"}],
    }
    fixture_path = manifest_root / "v1-baseline" / "test.jsonl"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text(json.dumps(fixture, sort_keys=True) + "\n", encoding="utf-8")
    fixture_sha = sha256(fixture_path)

    distributions = {
        "v1-baseline": ("uniform", 5303, 1),
        "v2-term-balance": ("term_balance", 2890, 2414),
        "v3-replay": ("replay", 4050, 1254),
    }
    lock = {
        "status": "passed",
        "seed": 42,
        "batch_size": 8,
        "builder_sha256": "b" * 64,
        "config_sha256": "d" * 64,
        "source_manifest_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
            "test": fixture_sha,
        },
        "shared_evaluation_manifests": True,
        "source_audit": {
            "status": "passed",
            "split_counts": {"train": 5303, "validation": 663, "test": 662},
            "unique_audio_files": 6628,
            "audio_verification_failures": [],
            "leakage_intersections": {
                field: {
                    "train_validation": 0,
                    "train_test": 0,
                    "validation_test": 0,
                }
                for field in (
                    "sample_id",
                    "audio_sha256",
                    "leakage_group_id",
                    "normalized_transcript",
                )
            },
        },
        "variants": {
            variant: {
                "strategy": distributions[variant][0],
                "manifest_sha256": {"test": fixture_sha},
                "distribution": {
                    "scheduled_rows": 5304,
                    "unique_source_rows": distributions[variant][1],
                    "repeated_exposures": distributions[variant][2],
                    "rows_with_terms": 2500,
                    "duration_hours": 6.5,
                    "unique_speakers": 10,
                    "maximum_speaker_share": 0.08,
                    "source_pool_counts": {"synthetic": 5304},
                    "term_category_occurrences": {"anatomy": 2500},
                },
            }
            for variant in VARIANTS
        },
    }
    lock_path = manifest_root / "dataset-lock.json"
    write_json(lock_path, lock)
    lock_sha = sha256(lock_path)

    shared_training = {
        "bf16": True,
        "physical_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 32,
        "max_steps": 1000,
        "nominal_epochs": 6,
        "learning_rate": 1e-5,
        "warmup_steps": 100,
        "evaluation_steps": 125,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
    }
    shared_models = {
        "tts_id": "microsoft/speecht5_tts",
        "tts_revision": "tts-revision",
        "vocoder_id": "microsoft/speecht5_hifigan",
        "vocoder_revision": "vocoder-revision",
        "speaker_encoder_id": "speechbrain/spkrec-xvect-voxceleb",
        "speaker_encoder_revision": "speaker-revision",
        "asr_id": "openai/whisper-small.en",
        "asr_revision": "asr-revision",
    }
    audit_variants = {}
    steps = list(range(125, 1001, 125))
    best_steps = (1000, 625, 1000)
    for index, variant in enumerate(VARIANTS):
        root = artifact_root / variant
        model = root / "selected-model" / "model.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(f"{variant}-weights".encode())
        artifact_manifest_path = root / "run_artifact_manifest.json"
        write_json(
            artifact_manifest_path,
            {
                "files": [
                    {
                        "path": "selected-model/model.safetensors",
                        "bytes": model.stat().st_size,
                        "sha256": sha256(model),
                    }
                ]
            },
        )
        write_json(
            root / "run_provenance.json",
            {
                "variant": variant,
                "training": shared_training,
                "models": shared_models,
                "configuration_sha256": "c" * 64,
                "dataset_lock_sha256": lock_sha,
                "dataset_lock_status": "passed",
                "stability_gate_decision": "user_authorized_skip",
                "early_stopping": {"patience_evaluations": 3, "threshold": 0.001},
            },
        )
        write_json(
            root / "training_metadata.json",
            {
                "log_history": [
                    {"step": step, "epoch": step / 166, "eval_loss": 0.6 - step / 10000}
                    for step in steps
                ],
                "train_metrics": {"train_runtime": 300 + index, "train_steps_per_second": 3.3},
            },
        )
        write_json(
            root / "evaluation" / "summary.json",
            {
                "case_count": 20,
                "failure_count": 0,
                "domain_term_accuracy": {
                    "correct": 2 + index,
                    "total": 4,
                    "accuracy": (2 + index) / 4,
                },
                "wer": 0.3 - index * 0.05,
                "average_inference_ms": 250 + index,
                "average_rtf": 0.15 + index * 0.01,
                "peak_gpu_memory_mb": 6000 + index,
                "synthesis_verification": {"status": "passed"},
            },
        )
        audit_variants[variant] = {
            "status": "completed",
            "pod_terminated": True,
            "pod_id": f"pod-{index}",
            "pod_created_utc": "2026-08-27T00:00:00Z",
            "pod_terminated_utc": "2026-08-27T01:00:00Z",
            "global_step": 1000,
            "best_step": best_steps[index],
            "best_eval_loss": 0.44 + index * 0.001,
            "stopped_early": False,
            "checkpoints": [{"step": step} for step in steps],
            "estimated_cost_usd": 0.5,
            "final_artifacts": {"manifest_sha256": sha256(artifact_manifest_path)},
        }

    pretrained_root = artifact_root / "pretrained"
    write_json(
        pretrained_root / "run_provenance.json",
        {
            "role": "pretrained-control",
            "dataset_lock_status": "passed",
            "dataset_lock_sha256": lock_sha,
            "test_manifest_sha256": fixture_sha,
            "test_case_count": 662,
            "models": shared_models,
            "pod_id": "pod-pretrained",
            "completed_utc": "2026-08-27T02:00:00Z",
            "environment": {"gpu": "NVIDIA GeForce RTX 4090"},
        },
    )
    write_json(
        pretrained_root / "evaluation" / "summary.json",
        {
            "case_count": 662,
            "failure_count": 0,
            "domain_term_accuracy": {"correct": 3, "total": 4, "accuracy": 0.75},
            "wer": 0.2,
            "average_inference_ms": 240,
            "average_rtf": 0.14,
            "peak_gpu_memory_mb": 5900,
            "synthesis_verification": {"status": "passed"},
        },
    )
    pretrained_files = [
        pretrained_root / "run_provenance.json",
        pretrained_root / "evaluation" / "summary.json",
    ]
    pretrained_manifest = pretrained_root / "evaluation_artifact_manifest.json"
    write_json(
        pretrained_manifest,
        {
            "files": [
                {
                    "path": path.relative_to(pretrained_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in pretrained_files
            ]
        },
    )
    write_json(
        pretrained_root / "EVALUATION_COMPLETE.json",
        {
            "status": "completed",
            "evaluation_artifact_manifest_sha256": sha256(pretrained_manifest),
        },
    )

    write_json(
        artifact_root / "final-audit.json",
        {
            "status": "passed",
            "run_id": "synthetic-stage11",
            "all_checkpoint_hashes_verified": True,
            "all_final_artifact_hashes_verified": True,
            "all_pods_terminated": True,
            "checkpoint_count": 24,
            "stability_gate_decision": "user_authorized_skip",
            "controller_incidents": [],
            "training_resumptions": [],
            "total_gpu_hours": 1.5,
            "estimated_total_cost_usd": 1.5,
            "variants": audit_variants,
        },
    )
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    pretrained_model = pretrained / "model.safetensors"
    pretrained_model.write_bytes(b"pretrained-weights")
    write_json(
        pretrained / "openvoice-model-manifest.json",
        {"weight_sha256": sha256(pretrained_model)},
    )
    return {
        "artifacts": artifact_root,
        "manifests": manifest_root,
        "pretrained": pretrained,
    }


class FakeExperimentRuntime:
    """Small deterministic CPU boundary for job orchestration tests."""

    def prepare(self) -> None:
        return None

    def synthesize(self, definition, text: str, output_path: Path):
        del definition, text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(np.zeros(1_600, dtype=np.int16).tobytes())
        return SpeechT5SynthesisOutput(
            audio_path=output_path,
            model_load_ms=25,
            inference_ms=50,
            audio_duration_ms=100,
            real_time_factor=0.5,
            process_memory_mb=512,
            warm=False,
        )

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        assert audio_path.is_file()
        return "arm pain", 20


class BlockingExperimentRuntime(FakeExperimentRuntime):
    """Pause synthesis so cancellation can be asserted at the thread boundary."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def synthesize(self, definition, text: str, output_path: Path):
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("Test synthesis was not released.")
        return super().synthesize(definition, text, output_path)


def make_job_service(
    tmp_path: Path,
    runtime: FakeExperimentRuntime,
    manifest_root: Path,
) -> tuple[ExperimentJobService, ExperimentJobStore]:
    source = tmp_path / "model"
    source.mkdir(exist_ok=True)
    (source / "model.safetensors").write_bytes(b"weights")
    models = ExperimentModelRegistry(
        (
            ExperimentModelDefinition(
                id="speecht5-pretrained",
                name="Pretrained",
                role="pretrained",
                variant="pretrained",
                source=source,
                revision="revision",
                model_sha256="a" * 64,
            ),
            ExperimentModelDefinition(
                id="speecht5-v3-replay",
                name="V3",
                role="adapted",
                variant="v3-replay",
                source=source,
                revision="revision",
                model_sha256="b" * 64,
            ),
        )
    )
    profile = tmp_path / "speaker-profile.json"
    profile.write_text(json.dumps({"embedding_sha256": "c" * 64}), encoding="utf-8")
    store = ExperimentJobStore(tmp_path / "comparisons")
    service = ExperimentJobService(
        fixtures=ExperimentFixtureService(manifest_root),
        models=models,
        runtime=runtime,
        text_processing=TextProcessingService(),
        store=store,
        audio_url_prefix="/experiment-audio",
        vocoder_revision="vocoder-revision",
        speaker_profile_path=profile,
    )
    return service, store


def test_report_is_derived_from_verified_stage11_artifacts(
    experiment_evidence: dict[str, Path],
) -> None:
    report = ExperimentReportService(
        experiment_evidence["artifacts"], experiment_evidence["manifests"]
    ).get()

    assert report.integrity.checkpoint_count == 24
    assert report.integrity.all_pods_terminated is True
    assert report.training.effective_batch_size == 32
    assert report.shared_splits == {"train": 5303, "validation": 663, "test": 662}
    assert report.data_audit.unique_audio_files == 6628
    assert report.data_audit.leakage_intersection_count == 0
    assert report.data_audit.schedule_block_size == 8
    assert report.pretrained_control.evaluation.case_count == 662
    assert report.pretrained_control.evaluation.domain_term_accuracy == 0.75
    assert report.pretrained_control.hardware == "NVIDIA GeForce RTX 4090"
    assert [item.best_step for item in report.variants] == [1000, 625, 1000]
    assert report.variants[2].evaluation.domain_terms_correct == 4
    assert report.variants[2].evaluation.domain_terms_total == 4
    assert report.variants[2].evaluation.domain_term_accuracy == 1
    assert len(report.variants[0].validation_history) == 8


def test_report_fails_closed_when_selected_model_is_tampered(
    experiment_evidence: dict[str, Path],
) -> None:
    model = (
        experiment_evidence["artifacts"]
        / "v2-term-balance"
        / "selected-model"
        / "model.safetensors"
    )
    model.write_bytes(b"tampered-but-same-size"[: model.stat().st_size])

    with pytest.raises(ExperimentEvidenceError, match="Selected model evidence is incomplete"):
        ExperimentReportService(
            experiment_evidence["artifacts"], experiment_evidence["manifests"]
        ).get()


def test_report_fails_closed_when_pretrained_evidence_is_tampered(
    experiment_evidence: dict[str, Path],
) -> None:
    summary = (
        experiment_evidence["artifacts"]
        / "pretrained"
        / "evaluation"
        / "summary.json"
    )
    summary.write_text(
        summary.read_text(encoding="utf-8").replace("240", "241", 1),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentEvidenceError,
        match="Pretrained evaluation artifact verification failed",
    ):
        ExperimentReportService(
            experiment_evidence["artifacts"], experiment_evidence["manifests"]
        ).get()


@pytest.mark.integration
@pytest.mark.skipif(
    not (ARTIFACT_ROOT / "final-audit.json").is_file()
    or not (MANIFEST_ROOT / "dataset-lock.json").is_file()
    or not (ARTIFACT_ROOT / "pretrained" / "EVALUATION_COMPLETE.json").is_file(),
    reason="Local ignored Stage 11 evidence is not provisioned.",
)
def test_local_report_matches_completed_stage11_run() -> None:
    report = ExperimentReportService(ARTIFACT_ROOT, MANIFEST_ROOT).get()

    assert report.integrity.checkpoint_count == 24
    assert report.pretrained_control.evaluation.case_count == 662
    assert report.pretrained_control.evaluation.domain_terms_correct == 381
    assert report.pretrained_control.evaluation.domain_terms_total == 416
    assert report.pretrained_control.evaluation.failure_count == 0
    assert report.pretrained_control.evaluation.synthesis_verified is True
    assert report.headline.startswith("SpeechT5 Pretrained led both")
    assert [item.best_step for item in report.variants] == [1000, 625, 1000]
    assert report.variants[2].evaluation.domain_terms_correct == 146
    assert report.variants[2].evaluation.domain_terms_total == 416


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        (APPROACH_RUN_ROOT / run_id / "final" / "RUN_COMPLETE.json").is_file()
        for run_id in (
            "v1a-real-20260828-001",
            "v1b-real-20260828-001",
            "v1c-real-20260828-001",
            "v1d-real-20260828-001",
        )
    ),
    reason="Local ignored V1 approach evidence is not provisioned.",
)
def test_local_v1_approach_report_matches_completed_runs() -> None:
    report = V1ApproachReportService(
        APPROACH_RUN_ROOT,
        ARTIFACT_ROOT,
        MANIFEST_ROOT,
    ).get()

    assert report.integrity.checkpoint_count == 40
    assert report.integrity.all_pods_terminated is True
    assert [item.id for item in report.variants] == [
        "v1a-conservative-full",
        "v1b-lora",
        "v1c-gradual-unfreeze",
        "v1d-reduction-factor-1",
    ]
    assert [item.selected_step for item in report.variants] == [25, 25, 25, 200]
    assert report.variants[2].evaluation.domain_terms_correct == 381
    assert report.variants[2].evaluation.domain_terms_total == 416
    assert report.variants[3].evaluation.failure_count == 68


def test_fixture_catalog_uses_locked_shared_manifest(
    experiment_evidence: dict[str, Path],
) -> None:
    fixtures = ExperimentFixtureService(experiment_evidence["manifests"])

    page = fixtures.list(term="arm", limit=5)

    assert page.total > 0
    assert len(page.items) <= 5
    assert all(item.target_terms for item in page.items)
    assert len(page.manifest_sha256) == 64
    assert fixtures.get(page.items[0].id).id == page.items[0].id


def test_term_scoring_and_wer_are_exact() -> None:
    score = score_terms(
        ["amlodipine", "hypertension"],
        "Amlodipine was spoken but the second term was missed.",
    )

    assert score.correct == ["amlodipine"]
    assert score.incorrect == ["hypertension"]
    assert score.accuracy == 0.5
    assert word_error_rate("the patient is stable", "the patient was stable") == 0.25
    assert word_error_rate("one two", "one two extra") == 0.5


def test_custom_request_requires_terms_present_in_text() -> None:
    with pytest.raises(ValidationError, match="must appear"):
        ExperimentComparisonRequest(
            mode="custom",
            text="The patient is stable.",
            target_terms=["amlodipine"],
            model_ids=["speecht5-pretrained", "speecht5-v3-replay"],
        )


def test_persistent_job_produces_partial_audio_then_scored_results(
    tmp_path: Path, experiment_evidence: dict[str, Path]
) -> None:
    async def run():
        service, store = make_job_service(
            tmp_path, FakeExperimentRuntime(), experiment_evidence["manifests"]
        )
        started = await service.start(
            ExperimentComparisonRequest(
                mode="custom",
                text="Arm pain",
                target_terms=["arm"],
                model_ids=["speecht5-pretrained", "speecht5-v3-replay"],
            )
        )
        for _ in range(100):
            job = service.get(started.id)
            if job.stage in {"completed", "completed_with_failures", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert job.stage == "completed"
        assert job.progress_percent == 100
        assert [result.status for result in job.results] == ["success", "success"]
        assert all(result.audio_url for result in job.results)
        assert all(result.target_terms and result.target_terms.accuracy == 1 for result in job.results)
        assert (store.directory(job.id) / "manifest.json").is_file()
        assert ExperimentJobStore(tmp_path / "comparisons").get(job.id).stage == "completed"

    asyncio.run(run())


def test_cancellation_survives_an_in_flight_cpu_operation(
    tmp_path: Path, experiment_evidence: dict[str, Path]
) -> None:
    async def run() -> None:
        runtime = BlockingExperimentRuntime()
        service, store = make_job_service(
            tmp_path, runtime, experiment_evidence["manifests"]
        )
        started = await service.start(
            ExperimentComparisonRequest(
                mode="custom",
                text="Arm pain",
                target_terms=["arm"],
                model_ids=["speecht5-pretrained", "speecht5-v3-replay"],
            )
        )
        assert await asyncio.to_thread(runtime.started.wait, 1)
        cancelled = service.cancel(started.id)
        runtime.release.set()
        await asyncio.sleep(0.05)

        assert cancelled.stage == "cancelled"
        assert service.get(started.id).stage == "cancelled"
        manifest = json.loads(
            (store.directory(started.id) / "manifest.json").read_text(encoding="utf-8")
        )
        assert any(item["path"].endswith(".wav") for item in manifest["files"])

    asyncio.run(run())


def test_recovery_completes_a_durably_queued_job(
    tmp_path: Path, experiment_evidence: dict[str, Path]
) -> None:
    request = ExperimentComparisonRequest(
        mode="custom",
        text="Arm pain",
        target_terms=["arm"],
        model_ids=["speecht5-pretrained", "speecht5-v3-replay"],
    )

    async def create_then_stop() -> str:
        service, _ = make_job_service(
            tmp_path, FakeExperimentRuntime(), experiment_evidence["manifests"]
        )
        return (await service.start(request)).id

    job_id = asyncio.run(create_then_stop())
    assert ExperimentJobStore(tmp_path / "comparisons").get(job_id).stage not in {
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
    }

    async def recover() -> None:
        service, _ = make_job_service(
            tmp_path, FakeExperimentRuntime(), experiment_evidence["manifests"]
        )
        await service.recover()
        for _ in range(100):
            job = service.get(job_id)
            if job.stage in {"completed", "completed_with_failures", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert job.stage == "completed"
        assert all(result.status == "success" for result in job.results)

    asyncio.run(recover())


def test_report_and_fixture_http_contracts(experiment_evidence: dict[str, Path]) -> None:
    reports = ExperimentReportService(
        experiment_evidence["artifacts"], experiment_evidence["manifests"]
    )
    fixtures = ExperimentFixtureService(experiment_evidence["manifests"])
    models = ExperimentModelRegistry.from_artifacts(
        experiment_evidence["artifacts"],
        experiment_evidence["pretrained"],
        "30fcde30f19b87502b8435427b5f5068e401d5f6",
    )
    service = ExperimentService(reports=reports, fixtures=fixtures, models=models, jobs=None)
    app = FastAPI()
    app.include_router(create_router(service), prefix="/api")

    async def send(path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    report = asyncio.run(send("/api/experiments/stage11/report"))
    fixture_page = asyncio.run(send("/api/experiments/stage11/fixtures?term=arm&limit=2"))
    model_page = asyncio.run(send("/api/experiments/stage11/models"))

    assert report.status_code == 200
    assert report.json()["integrity"]["checkpointCount"] == 24
    assert fixture_page.status_code == 200
    assert len(fixture_page.json()["items"]) == 1
    assert model_page.status_code == 200
    assert len(model_page.json()) == 4
