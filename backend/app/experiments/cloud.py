"""Compose live cloud comparisons from snapshot evidence and remote model artifacts."""

from importlib.util import find_spec
from pathlib import Path

from app.experiments.common import atomic_json, sha256_file
from app.experiments.jobs import ExperimentJobService
from app.experiments.model_registry import ExperimentModelRegistry
from app.experiments.remote_artifacts import (
    ExperimentModelProvisioner,
    RemoteModelCatalog,
)
from app.experiments.snapshot import DEFAULT_SNAPSHOT_ROOT, SnapshotExperimentService
from app.experiments.store import ExperimentJobStore
from app.inference.speecht5_cpu import SpeechT5CpuRuntime
from app.scheduling.service import ProcessingScheduler
from app.text_processing.service import TextProcessingService


def create_cloud_experiment_service(
    *,
    model_base_url: str | None,
    model_access_token: str | None,
    remote_manifest_path: Path,
    pretrained_root: Path,
    vocoder_root: Path,
    asr_root: Path,
    speaker_embedding_path: Path,
    stage12_root: Path,
    remote_cache_root: Path,
    text_processing: TextProcessingService,
    audio_url_prefix: str,
    vocoder_revision: str,
    maximum_queued_jobs: int,
    maximum_cached_models: int,
    cpu_threads: int | None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    scheduler: ProcessingScheduler | None = None,
) -> SnapshotExperimentService:
    """Enable live jobs only when every shared dependency and remote source is real."""
    snapshot = SnapshotExperimentService(snapshot_root)
    if not model_base_url:
        return snapshot

    required_modules = ("soundfile", "torch", "transformers")
    dependencies_ready = all(find_spec(module) is not None for module in required_modules)
    required_paths = (
        pretrained_root / "pytorch_model.bin",
        vocoder_root / "pytorch_model.bin",
        asr_root / "model.safetensors",
        speaker_embedding_path,
        remote_manifest_path,
    )
    if not dependencies_ready or not all(path.is_file() for path in required_paths):
        return snapshot

    catalog = RemoteModelCatalog.from_path(remote_manifest_path)
    models = ExperimentModelRegistry.from_snapshot_models(
        snapshot.source_models(),
        pretrained_root=pretrained_root,
        remote_cache_root=remote_cache_root,
        remote_model_ids=catalog.model_ids,
    )
    if not all(model.available and model.model_sha256 for model in models.list()):
        return snapshot

    profile_root = stage12_root / "serving-profile"
    profile_path = profile_root / "speaker-profile.json"
    embedding_sha256 = sha256_file(speaker_embedding_path)
    atomic_json(
        profile_path,
        {
            "schema_version": 1,
            "source": "pinned-product-speaker-profile",
            "embedding_sha256": embedding_sha256,
        },
    )
    runtime = SpeechT5CpuRuntime(
        vocoder_root=vocoder_root,
        asr_root=asr_root,
        speaker_embedding_path=speaker_embedding_path,
        maximum_cached_models=maximum_cached_models,
        cpu_threads=cpu_threads,
    )
    provisioner = ExperimentModelProvisioner(
        base_url=model_base_url,
        catalog=catalog,
        bearer_token=model_access_token,
    )
    jobs = ExperimentJobService(
        fixtures=snapshot.fixture_catalog,
        models=models,
        runtime=runtime,
        text_processing=text_processing,
        store=ExperimentJobStore(stage12_root / "comparisons"),
        audio_url_prefix=audio_url_prefix,
        vocoder_revision=vocoder_revision,
        speaker_profile_path=profile_path,
        maximum_queued_jobs=maximum_queued_jobs,
        provisioner=provisioner,
        scheduler=scheduler,
    )
    return SnapshotExperimentService(snapshot_root, models=models, jobs=jobs)
