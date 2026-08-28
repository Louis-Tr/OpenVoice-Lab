from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def configure_reduction_factor(model: Any, config: dict[str, Any]) -> int:
    """Apply the experiment reduction factor before collation and training."""

    expected = int(
        config["training"].get("reduction_factor", model.config.reduction_factor)
    )
    if expected <= 0:
        raise ValueError("training.reduction_factor must be positive")
    model.config.reduction_factor = expected
    if int(model.config.reduction_factor) != expected:
        raise RuntimeError("SpeechT5 reduction factor was not applied")
    return expected


def assert_saved_reduction_factor(path: Path, expected: int) -> None:
    """Verify that a checkpoint/export persisted the expected SpeechT5 config."""

    config_path = path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"saved SpeechT5 config is missing: {config_path}")
    values = json.loads(config_path.read_text(encoding="utf-8"))
    observed = int(values.get("reduction_factor", -1))
    if observed != int(expected):
        raise RuntimeError(
            f"saved SpeechT5 reduction factor mismatch: expected {expected}, got {observed}"
        )


def assert_loaded_reduction_factor(model: Any, expected: int, source: Path) -> None:
    """Verify that an actual loaded generation model retained the experiment config."""

    observed = int(model.config.reduction_factor)
    if observed != int(expected):
        raise RuntimeError(
            f"loaded SpeechT5 reduction factor mismatch from {source}: "
            f"expected {expected}, got {observed}"
        )
