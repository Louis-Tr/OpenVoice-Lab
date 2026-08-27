"""Atomic filesystem persistence for live comparison jobs."""

from pathlib import Path
from threading import Lock

from app.experiments.common import ExperimentJobNotFoundError, atomic_json, read_json, sha256_file
from app.schemas.experiment import ExperimentComparisonJob

TERMINAL_STAGES = {"completed", "completed_with_failures", "failed", "cancelled"}


class ExperimentJobStore:
    """Persist every public job snapshot so browser refreshes are safe."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def create(self, job: ExperimentComparisonJob) -> None:
        directory = self.directory(job.id)
        with self._lock:
            if directory.exists():
                raise ValueError(f"Experiment job already exists: {job.id}")
            directory.mkdir(parents=True)
            atomic_json(directory / "request.json", job.model_dump(mode="json", by_alias=True))
            self._write(job)

    def save(self, job: ExperimentComparisonJob) -> None:
        with self._lock:
            self._write(job)

    def get(self, job_id: str) -> ExperimentComparisonJob:
        path = self.directory(job_id) / "status.json"
        if not path.is_file():
            raise ExperimentJobNotFoundError(f"Experiment job '{job_id}' was not found.")
        return ExperimentComparisonJob.model_validate(read_json(path))

    def nonterminal(self) -> list[ExperimentComparisonJob]:
        jobs = []
        for path in self.root.glob("*/status.json"):
            job = ExperimentComparisonJob.model_validate(read_json(path))
            if job.stage not in TERMINAL_STAGES:
                jobs.append(job)
        return jobs

    def directory(self, job_id: str) -> Path:
        if not job_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in job_id):
            raise ExperimentJobNotFoundError(f"Experiment job '{job_id}' was not found.")
        return self.root / job_id

    def write_manifest(self, job_id: str) -> Path:
        directory = self.directory(job_id)
        entries = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                entries.append(
                    {
                        "path": path.relative_to(directory).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = directory / "manifest.json"
        atomic_json(manifest, {"schema_version": 1, "files": entries})
        return manifest

    def _write(self, job: ExperimentComparisonJob) -> None:
        directory = self.directory(job.id)
        payload = job.model_dump(mode="json", by_alias=True)
        atomic_json(directory / "status.json", payload)
        if job.stage in TERMINAL_STAGES:
            atomic_json(directory / "result.json", payload)
