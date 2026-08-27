"""Locked Stage 11 fixture catalog."""

from __future__ import annotations

import json
from pathlib import Path

from app.experiments.common import (
    ExperimentEvidenceError,
    ExperimentFixtureNotFoundError,
    read_json,
    sha256_file,
)
from app.schemas.experiment import ExperimentFixture, ExperimentFixturePage, ExperimentFixtureTerm


class ExperimentFixtureService:
    """Serve text and target terms from the verified shared test manifest."""

    def __init__(self, manifest_root: Path) -> None:
        self._root = manifest_root.resolve()
        self._manifest_path = self._root / "v1-baseline" / "test.jsonl"
        lock = read_json(self._root / "dataset-lock.json")
        hashes = {
            item["manifest_sha256"]["test"] for item in lock.get("variants", {}).values()
        }
        if lock.get("status") != "passed" or len(hashes) != 1:
            raise ExperimentEvidenceError("The shared Stage 11 fixture manifest is not locked.")
        self._manifest_sha256 = hashes.pop()
        if not self._manifest_path.is_file() or sha256_file(self._manifest_path) != self._manifest_sha256:
            raise ExperimentEvidenceError("The Stage 11 fixture manifest hash does not match its lock.")
        self._fixtures = self._load()
        self._by_id = {fixture.id: fixture for fixture in self._fixtures}

    def list(
        self,
        *,
        query: str | None = None,
        term: str | None = None,
        category: str | None = None,
        offset: int = 0,
        limit: int = 30,
    ) -> ExperimentFixturePage:
        selected = self._fixtures
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
                if any(candidate.category.casefold() == needle for candidate in item.target_terms)
            ]
        return ExperimentFixturePage(
            items=selected[offset : offset + limit],
            total=len(selected),
            offset=offset,
            limit=limit,
            manifest_sha256=self._manifest_sha256,
        )

    def get(self, fixture_id: str) -> ExperimentFixture:
        try:
            return self._by_id[fixture_id]
        except KeyError as error:
            raise ExperimentFixtureNotFoundError(
                f"Experiment fixture '{fixture_id}' was not found."
            ) from error

    def _load(self) -> list[ExperimentFixture]:
        fixtures = []
        with self._manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExperimentEvidenceError(
                        f"Invalid fixture JSON on line {line_number}."
                    ) from error
                seen: set[tuple[str, str]] = set()
                terms = []
                for item in row.get("medical_terms", []):
                    key = (item["canonical"].casefold(), item["category"].casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    terms.append(
                        ExperimentFixtureTerm(
                            text=item["text"],
                            canonical=item["canonical"],
                            category=item["category"],
                        )
                    )
                if not terms:
                    continue
                fixtures.append(
                    ExperimentFixture(id=row["sample_id"], text=row["text"], target_terms=terms)
                )
        return fixtures
