"""Hash-verified read-only experiment evidence for lightweight deployments."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from app.experiments.common import ExperimentEvidenceError, read_json, sha256_file
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


class SnapshotExperimentService(ExperimentService):
    """Expose verified aggregate evidence when live model artifacts are absent."""

    def __init__(self, root: Path = DEFAULT_SNAPSHOT_ROOT) -> None:
        self._root = root.resolve()
        manifest = read_json(self._root / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ExperimentEvidenceError("The experiment snapshot schema is unsupported.")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ExperimentEvidenceError("The experiment snapshot manifest is incomplete.")
        for filename in ("report.json", "fixtures.json", "models.json"):
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
        self._fixtures = fixture_page.items
        self._fixture_manifest_sha256 = fixture_page.manifest_sha256
        models = TypeAdapter(list[ExperimentModelSummary]).validate_json(
            (self._root / "models.json").read_text(encoding="utf-8")
        )
        self._models = [
            model.model_copy(
                update={
                    "available": False,
                    "unavailable_reason": (
                        "Historical metrics are available in this cloud image; live SpeechT5 "
                        "weights are not packaged."
                    ),
                }
            )
            for model in models
        ]

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
        selected: list[ExperimentFixture] = self._fixtures
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
            manifest_sha256=self._fixture_manifest_sha256,
        )

    def models(self) -> list[ExperimentModelSummary]:
        return self._models

    async def start(
        self, request: ExperimentComparisonRequest
    ) -> ExperimentComparisonJob:
        del request
        raise self._live_unavailable()

    def get(self, job_id: str) -> ExperimentComparisonJob:
        del job_id
        raise self._live_unavailable()

    def cancel(self, job_id: str) -> ExperimentComparisonJob:
        del job_id
        raise self._live_unavailable()

    async def recover(self) -> None:
        return None

    @staticmethod
    def _live_unavailable() -> ExperimentEvidenceError:
        return ExperimentEvidenceError(
            "This deployment includes verified historical evidence but not the live "
            "SpeechT5 model runtime."
        )
