"""CPU pool configuration and per-request runtime state."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.inference import cpu, kokoro_onnx, speecht5_random


def test_torch_configuration_once_and_conflicts_fail(monkeypatch):
    monkeypatch.setattr(cpu, "_configured_threads", None)
    torch = SimpleNamespace(set_num_threads=Mock(), set_num_interop_threads=Mock())
    cpu.configure_torch_threads(torch, 2)
    cpu.configure_torch_threads(torch, 2)
    cpu.configure_torch_threads(torch, None)
    torch.set_num_threads.assert_called_once_with(2)
    torch.set_num_interop_threads.assert_called_once_with(1)
    with pytest.raises(RuntimeError, match="already 2"):
        cpu.configure_torch_threads(torch, 3)


def test_kokoro_session_controls_and_concurrent_runs(monkeypatch):
    barrier = Barrier(2)
    options = SimpleNamespace(add_session_config_entry=Mock())
    runtime = SimpleNamespace(
        get_voices=lambda: ["voice"], tokenizer=SimpleNamespace(phonemize=lambda text, lang: text)
    )

    def generate(text, **kwargs):
        assert kwargs["is_phonemes"]
        barrier.wait(5)
        return np.ones(len(text), dtype=np.float32), 24000

    runtime.create = generate
    session = Mock()
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(kokoro_onnx.ort, "SessionOptions", lambda: options)
    monkeypatch.setattr(kokoro_onnx.ort, "InferenceSession", session)
    monkeypatch.setattr(kokoro_onnx.Kokoro, "from_session", factory)
    engine = kokoro_onnx.KokoroONNXEngine(Path("model"), Path("voices"), cpu_threads=2)
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == kokoro_onnx.ort.ExecutionMode.ORT_SEQUENTIAL
    assert session.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    options.add_session_config_entry.assert_any_call("session.intra_op.allow_spinning", "0")
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(engine.synthesize, "one", "voice")
        b = pool.submit(engine.synthesize, "other", "voice")
        assert a.result(10).samples.size == 3
        assert b.result(10).samples.size == 5


def test_speecht5_dropout_matches_serial_seed_and_is_request_local(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(speecht5_random, "version", lambda package: "4.57.6")

    class Prenet:
        def _consistent_dropout(self, inputs_embeds, p):
            mask = torch.bernoulli(inputs_embeds[0], p=p)
            all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
            return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    prenet = Prenet()
    inputs = torch.ones(1, 10, 10)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = [prenet._consistent_dropout(inputs, 0.5) for _ in range(3)]
    model = SimpleNamespace(speecht5=SimpleNamespace(decoder=SimpleNamespace(prenet=prenet)))
    speecht5_random.install_request_dropout(model, torch)
    barrier = Barrier(2)

    def run():
        token = speecht5_random._generator.set(torch.Generator(device="cpu").manual_seed(42))
        try:
            barrier.wait(5)
            actual = [prenet._consistent_dropout(inputs, 0.5) for _ in range(3)]
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        finally:
            speecht5_random._generator.reset(token)
        with pytest.raises(LookupError):
            speecht5_random._generator.get()

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(run), pool.submit(run)
        a.result(10)
        b.result(10)


def test_unsupported_speecht5_version_fails(monkeypatch):
    monkeypatch.setattr(speecht5_random, "version", lambda package: "different")
    with pytest.raises(RuntimeError, match="pinned"):
        speecht5_random.install_request_dropout(None, None)
