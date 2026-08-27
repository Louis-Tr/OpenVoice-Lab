from __future__ import annotations

import math
from numbers import Real
from typing import Any


def speech_labels_for_example(labels: Any) -> Any:
    """Remove SpeechT5's one-item batch dimension with shape validation."""
    shape = getattr(labels, "shape", None)
    if shape is not None:
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError(
                "SpeechT5 labels must have shape [1, frames, mel_bins]; "
                f"received {tuple(shape)}"
            )
        return labels[0]
    if not isinstance(labels, (list, tuple)) or len(labels) != 1:
        raise ValueError("SpeechT5 labels must contain exactly one batched example")
    example = labels[0]
    if not isinstance(example, (list, tuple)) or not example:
        raise ValueError("SpeechT5 labels contain no acoustic frames")
    return example


def validate_training_safety(training: dict[str, Any], *, bf16_supported: bool) -> None:
    fp16 = bool(training.get("fp16", False))
    bf16 = bool(training.get("bf16", False))
    if fp16 == bf16:
        raise ValueError(
            "Exactly one of training.fp16 and training.bf16 must be enabled"
        )
    if bf16 and not bf16_supported:
        raise RuntimeError("BF16 training is configured but unsupported by this GPU")
    if float(training.get("max_grad_norm", 0)) <= 0:
        raise ValueError("training.max_grad_norm must be positive")
    if int(training.get("checkpoint_steps", 0)) <= 0:
        raise ValueError("training.checkpoint_steps must be positive")
    if int(training.get("evaluation_steps", 0)) <= 0:
        raise ValueError("training.evaluation_steps must be positive")


def non_finite_training_values(logs: dict[str, Any]) -> dict[str, float]:
    monitored = ("loss", "eval_loss", "grad_norm")
    return {
        name: float(logs[name])
        for name in monitored
        if isinstance(logs.get(name), Real) and not math.isfinite(float(logs[name]))
    }


def require_successful_comparison(summary: dict[str, Any]) -> None:
    failures = {
        candidate: int(summary[candidate]["failure_count"])
        for candidate in ("base", "adapted")
        if int(summary[candidate]["failure_count"]) > 0
    }
    if failures:
        raise RuntimeError(
            f"Base/adapted comparison is incomplete; recorded failures: {failures}"
        )
