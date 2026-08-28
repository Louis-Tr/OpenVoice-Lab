from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.full_training.reduction_factor import (
    assert_loaded_reduction_factor,
    assert_saved_reduction_factor,
    configure_reduction_factor,
)


def _model(value: int) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(reduction_factor=value))


def test_reduction_factor_is_applied_before_collator_construction() -> None:
    model = _model(2)
    config = {"training": {"reduction_factor": 1}}

    configured = configure_reduction_factor(model, config)
    collator_value = model.config.reduction_factor

    assert configured == 1
    assert collator_value == 1


def test_checkpoint_and_final_export_persist_reduction_factor(tmp_path: Path) -> None:
    for name in ("checkpoint-25", "selected-model"):
        saved = tmp_path / name
        saved.mkdir()
        (saved / "config.json").write_text(
            json.dumps({"reduction_factor": 1}), encoding="utf-8"
        )

        assert_saved_reduction_factor(saved, 1)


def test_checkpoint_resume_rejects_wrong_reduction_factor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"reduction_factor": 2}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="expected 1, got 2"):
        assert_saved_reduction_factor(checkpoint, 1)


def test_generation_model_load_assertion_uses_expected_config(tmp_path: Path) -> None:
    assert_loaded_reduction_factor(_model(1), 1, tmp_path / "selected-model")

    with pytest.raises(RuntimeError, match="loaded SpeechT5 reduction factor mismatch"):
        assert_loaded_reduction_factor(_model(2), 1, tmp_path / "selected-model")
