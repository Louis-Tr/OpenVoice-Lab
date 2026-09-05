"""Hash-verified read-only experiment evidence for lightweight deployments."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from app.experiments.common import (
    ExperimentEvidenceError,
    ExperimentFixtureNotFoundError,
    read_json,
    sha256_file,
)
from app.experiments.jobs import ExperimentJobService
from app.experiments.model_registry import ExperimentModelRegistry
from app.experiments.service import ExperimentService
from app.schemas.experiment import (
    ExperimentComparisonJob,
    ExperimentComparisonRequest,
    ExperimentFixture,
    ExperimentFixturePage,
    ExperimentModelSummary,
    ExperimentReport,
)

DEFAULT_SNAPSHOT_ROOT = Path(__file__).with_name("snapshots")


class SnapshotFixtureCatalog:
    """Read-only fixture catalog backed by the verified cloud snapshot."""

    def __init__(self, items: list[ExperimentFixture], manifest_sha256: str) -> None:
        self._items = items
        self._by_id = {item.id: item for item in items}
        self._manifest_sha256 = manifest_sha256

    def get(self, fixture_id: str) -> ExperimentFixture:
        try:
            return self._by_id[fixture_id]
        except KeyError as error:
            raise ExperimentFixtureNotFoundError(
                f"Experiment fixture '{fixture_id}' was not found."
            ) from error

    def list(
        self,
        *,
        query: str | None,
        term: str | None,
        category: str | None,
        offset: int,
        limit: int,
    ) -> ExperimentFixturePage:
        selected = self._items
        if query:
            needle = query.casefold()
            selected = [item for item in selected if needle in item.text.casefold()]
        if term:
            needle = term.casefold()
            selected = [
                item
                for item in selected
                if any(
                    needle in candidate.text.casefold()
                    or needle in candidate.canonical.casefold()
                    for candidate in item.target_terms
                )
            ]
        if category:
            needle = category.casefold()
            selected = [
                item
                for item in selected
                if any(
                    candidate.category.casefold() == needle
                    for candidate in item.target_terms
                )
            ]
        return ExperimentFixturePage(
            items=selected[offset : offset + limit],
            total=len(selected),
            offset=offset,
            limit=limit,
            manifest_sha256=self._manifest_sha256,
        )


class SnapshotExperimentService(ExperimentService):
    """Expose verified aggregate evidence when live model artifacts are absent."""

    def __init__(
        self,
        root: Path = DEFAULT_SNAPSHOT_ROOT,
        *,
        models: ExperimentModelRegistry | None = None,
        jobs: ExperimentJobService | None = None,
    ) -> None:
        self._root = root.resolve()
        manifest = read_json(self._root / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ExperimentEvidenceError("The experiment snapshot schema is unsupported.")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ExperimentEvidenceError("The experiment snapshot manifest is incomplete.")
        for filename in (
            "report.json",
            "fixtures.json",
            "models.json",
            "remote_models.json",
        ):
            path = self._root / filename
            if not path.is_file() or sha256_file(path) != files.get(filename):
                raise ExperimentEvidenceError(
                    f"The experiment snapshot hash differs for {filename}."
                )

        self._report = ExperimentReport.model_validate_json(
            (self._root / "report.json").read_text(encoding="utf-8")
        )
        fixture_page = ExperimentFixturePage.model_validate_json(
            (self._root / "fixtures.json").read_text(encoding="utf-8")
        )
        self.fixture_catalog = SnapshotFixtureCatalog(
            fixture_page.items,
            fixture_page.manifest_sha256,
        )
        snapshot_models = TypeAdapter(list[ExperimentModelSummary]).validate_json(
            (self._root / "models.json").read_text(encoding="utf-8")
        )
        self._snapshot_models = snapshot_models
        self._jobs = jobs
        self._models = (
            models.list()
            if models is not None
            else [
                model.model_copy(
                    update={
                        "available": False,
                        "unavailable_reason": (
                            "Historical metrics are available in this cloud image; live "
                            "SpeechT5 weights are not provisioned."
                        ),
                    }
                )
                for model in snapshot_models
            ]
        )

    def report(self) -> ExperimentReport:
        return self._report

    def fixtures(
        self,
        *,
        query: str | None,
        term: str | None,
        category: str | None,
        offset: int,
        limit: int,
    ) -> ExperimentFixturePage:
        return self.fixture_catalog.list(
            query=query,
            term=term,
            category=category,
            offset=offset,
            limit=limit,
        )

    def models(self) -> list[ExperimentModelSummary]:
        return self._models

    def source_models(self) -> list[ExperimentModelSummary]:
        """Return verified immutable model identities before readiness decoration."""
        return self._snapshot_models

    async def start(
        self, request: ExperimentComparisonRequest
    ) -> ExperimentComparisonJob:
        if self._jobs is None:
            raise self._live_unavailable()
        return await self._jobs.start(request)

    def get(self, job_id: str) -> ExperimentComparisonJob:
        if self._jobs is None:
            raise self._live_unavailable()
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ExperimentComparisonJob:
        if self._jobs is None:
            raise self._live_unavailable()
        return self._jobs.cancel(job_id)

    async def recover(self) -> None:
        if self._jobs is not None:
            await self._jobs.recover()

    @staticmethod
    def _live_unavailable() -> ExperimentEvidenceError:
        return ExperimentEvidenceError(
            "This deployment includes verified historical evidence but not the live "
            "SpeechT5 model runtime."
        )
