"""Unit coverage for the product Audio8 INT4 adapter."""

from pathlib import Path

import numpy as np
import pytest

from app.inference.audio8_onnx import Audio8ONNXEngine
from app.inference.base import InferenceError, UnsupportedVoiceError


class FakeRuntime:
    def __init__(self, model_root: Path, threads: int | None) -> None:
        self.model_root = model_root
        self.threads = threads
        self.calls: list[tuple[str, int, int]] = []

    def synthesize(self, text: str, *, max_new_tokens: int, seed: int) -> np.ndarray:
        self.calls.append((text, max_new_tokens, seed))
        return np.linspace(-0.1, 0.1, 441, dtype=np.float32)


def test_audio8_adapter_uses_cpu_runtime_with_bounded_deterministic_generation(
    tmp_path: Path,
) -> None:
    runtime: FakeRuntime | None = None

    def create_runtime(model_root: Path, threads: int | None) -> FakeRuntime:
        nonlocal runtime
        runtime = FakeRuntime(model_root, threads)
        return runtime

    engine = Audio8ONNXEngine(
        tmp_path,
        threads=2,
        runtime_factory=create_runtime,
    )
    result = engine.synthesize("CPU synthesis works.", "unconditioned")

    assert runtime is not None
    assert runtime.model_root == tmp_path
    assert runtime.threads == 2
    assert runtime.calls == [("CPU synthesis works.", 160, 42)]
    assert engine.voices == ("unconditioned",)
    assert result.sample_rate_hz == 44_100
    assert result.samples.dtype == np.float32
    assert result.samples.size == 441


def test_audio8_adapter_rejects_unsupported_controls(tmp_path: Path) -> None:
    engine = Audio8ONNXEngine(tmp_path, runtime_factory=FakeRuntime)

    with pytest.raises(UnsupportedVoiceError):
        engine.synthesize("hello", "unknown")
    with pytest.raises(InferenceError, match="speed control"):
        engine.synthesize("hello", "unconditioned", speed=1.25)


def test_audio8_adapter_wraps_runtime_failures(tmp_path: Path) -> None:
    class FailedRuntime(FakeRuntime):
        def synthesize(self, text: str, *, max_new_tokens: int, seed: int) -> np.ndarray:
            raise RuntimeError("decoder failed")

    engine = Audio8ONNXEngine(tmp_path, runtime_factory=FailedRuntime)

    with pytest.raises(InferenceError, match="decoder failed"):
        engine.synthesize("hello", "unconditioned")
