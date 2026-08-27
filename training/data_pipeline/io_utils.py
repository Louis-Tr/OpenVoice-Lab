from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str, length: int | None = None) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def _atomic_path(path: Path) -> tuple[Path, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    return Path(temporary), os.fdopen(descriptor, "w", encoding="utf-8", newline="")


def atomic_write_text(path: Path, text: str) -> None:
    temporary, handle = _atomic_path(path)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary, handle = _atomic_path(path)
    try:
        with handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]
) -> None:
    temporary, handle = _atomic_path(path)
    try:
        with handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
