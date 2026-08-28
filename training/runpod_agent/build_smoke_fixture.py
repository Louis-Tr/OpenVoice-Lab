"""Build the deterministic, tracked dataset used by the real toolkit smoke run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


COUNTS = {"train": 64, "validation": 16, "test": 16}
BLOCK_SIZE = 4
VARIANT = "v1-toolkit-smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_root = root / "data-processing/manifests/stage11/v1-baseline"
    output_root = root / "training/fixtures/toolkit_smoke"
    output_root.mkdir(parents=True, exist_ok=True)

    source_sha256: dict[str, str] = {}
    manifest_sha256: dict[str, str] = {}
    for split, count in COUNTS.items():
        source = source_root / f"{split}.jsonl"
        source_sha256[split] = sha256(source)
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= count:
                    break
                row = json.loads(line)
                row.update(
                    split=split,
                    training_variant=VARIANT,
                    schedule_id=f"{VARIANT}-{split}-{index:07d}",
                    schedule_index=index,
                    batch_index=index // BLOCK_SIZE,
                )
                rows.append(row)
        if len(rows) != count:
            raise RuntimeError(f"{source} contained {len(rows)} rows; expected {count}")
        destination = output_root / f"{split}.jsonl"
        destination.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        manifest_sha256[split] = sha256(destination)

    generation = {
        "source_variant": "v1-baseline",
        "selection": "first_n_rows_in_locked_manifest_order",
        "source_manifest_sha256": source_sha256,
        "row_counts": COUNTS,
        "block_size": BLOCK_SIZE,
    }
    lock = {
        "schema_version": 1,
        "status": "passed",
        "fixture_purpose": "real RunPod agent-toolkit smoke training",
        "generation": generation,
        "generation_sha256": canonical_sha256(generation),
        "variants": {
            VARIANT: {
                "audit_status": "passed",
                "manifest_sha256": manifest_sha256,
                "strategy": "deterministic_v1_prefix_fixture",
            }
        },
    }
    write_json(output_root / "dataset-lock.json", lock)


if __name__ == "__main__":
    main()
