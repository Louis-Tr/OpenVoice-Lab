"""Sequential SpeechT5 generation without loading model weights."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.inference.base import InferenceError, InputTooLongError
from app.inference.speecht5 import SpeechT5InferenceEngine
from app.inference.speecht5_random import _generator
from app.inference.text_chunks import split_text


def fake_engine(limit=600):
    engine = SpeechT5InferenceEngine.__new__(SpeechT5InferenceEngine)
    engine._voices = ("cmu-slt",)
    engine._processor_lock = Lock()
    engine._torch = SimpleNamespace(Generator=Mock(), inference_mode=nullcontext)
    tokenizer = Mock(side_effect=lambda text, **kw: {"input_ids": [0] * (len(text) + 2)})
    tokenizer.model_max_length = limit
    processor = Mock(
        side_effect=lambda text, **kw: {
            "input_ids": np.zeros((1, len(text) + 2), dtype=np.int64),
            "attention_mask": np.ones((1, len(text) + 2), dtype=np.int64),
        }
    )
    processor.tokenizer = tokenizer
    engine._processor = processor
    waveform = Mock()
    waveform.detach().float().cpu().numpy.return_value = np.ones(160, dtype=np.float32)
    engine._model = SimpleNamespace(
        config=SimpleNamespace(max_text_positions=limit),
        generate_speech=Mock(return_value=waveform),
    )
    engine._speaker = object()
    engine._vocoder = object()

    return engine


@pytest.mark.parametrize("length, calls", [(598, 1), (599, 2), (641, 2), (5000, 9)])
def test_speecht5_chunks_at_actual_token_limit(length, calls):
    engine = fake_engine()
    result = engine.synthesize("a" * length, "cmu-slt")
    generate = engine._model.generate_speech
    assert generate.call_count == calls
    assert result.sample_rate_hz == 16_000
    assert result.samples.size == calls * 160
    texts = [call.kwargs["text"] for call in engine._processor.call_args_list]
    assert "".join(texts) == "a" * length
    for call in generate.call_args_list:
        assert call.args[0].shape[-1] <= 600
        assert call.kwargs["attention_mask"].shape == call.args[0].shape
        assert call.args[1] is engine._speaker
        assert call.kwargs["vocoder"] is engine._vocoder
    engine._torch.Generator.assert_called_once_with(device="cpu")
    engine._torch.Generator.return_value.manual_seed.assert_called_once_with(42)
    with pytest.raises(LookupError):
        _generator.get()


def test_speecht5_concatenates_in_order_with_one_generator():
    engine = fake_engine(limit=12)
    seen = []

    def generate(*args, **kwargs):
        seen.append(_generator.get())
        waveform = Mock()
        waveform.detach().float().cpu().numpy.return_value = np.full(2, len(seen))
        return waveform

    engine._model.generate_speech.side_effect = generate
    result = engine.synthesize("First. Second. Third.", "cmu-slt")
    np.testing.assert_array_equal(result.samples, [1, 1, 2, 2, 3, 3])
    assert all(generator is seen[0] for generator in seen)


def test_later_chunk_failure_resets_context_and_stops_generation():
    engine = fake_engine(limit=12)
    good = engine._model.generate_speech.return_value
    engine._model.generate_speech.side_effect = [good, RuntimeError("chunk failed"), good]
    with pytest.raises(InferenceError, match="chunk failed"):
        engine.synthesize("First. Second. Third.", "cmu-slt")
    assert engine._model.generate_speech.call_count == 2
    with pytest.raises(LookupError):
        _generator.get()


@pytest.mark.parametrize("values", [[], [float("nan")], [[1.0]]])
def test_invalid_chunk_stops_generation(values):
    engine = fake_engine(limit=12)
    waveform = engine._model.generate_speech.return_value
    waveform.detach().float().cpu().numpy.return_value = np.asarray(values)
    with pytest.raises(InferenceError, match="invalid audio"):
        engine.synthesize("First. Second. Third.", "cmu-slt")
    assert engine._model.generate_speech.call_count == 1
    with pytest.raises(LookupError):
        _generator.get()


def test_processor_token_limit_still_checked():
    engine = fake_engine()
    engine._processor.side_effect = lambda **kwargs: {"input_ids": np.zeros((1, 601))}
    with pytest.raises(InputTooLongError, match="chunk exceeds 600"):
        engine.synthesize("Short.", "cmu-slt")
    engine._model.generate_speech.assert_not_called()


def test_concurrent_chunked_requests_keep_generators_and_audio_separate():
    engine = fake_engine(limit=12)
    generators = [Mock(), Mock()]
    for generator in generators:
        generator.manual_seed.return_value = generator
    engine._torch.Generator.side_effect = generators
    barrier = Barrier(2)

    def generate(*args, **kwargs):
        identity = next(i + 1 for i, item in enumerate(generators) if item is _generator.get())
        barrier.wait(5)  # Both requests must overlap at every chunk.
        waveform = Mock()
        waveform.detach().float().cpu().numpy.return_value = np.full(2, identity)
        return waveform

    def run():
        result = engine.synthesize("First. Second. Third.", "cmu-slt")
        with pytest.raises(LookupError):
            _generator.get()
        return result.samples

    engine._model.generate_speech.side_effect = generate
    with ThreadPoolExecutor(2) as pool:
        first, second = pool.submit(run), pool.submit(run)
        results = [first.result(10), second.result(10)]
    assert sorted(float(samples[0]) for samples in results) == [1, 2]
    assert all(samples.size == 6 and np.all(samples == samples[0]) for samples in results)


@pytest.mark.parametrize(
    "text, limit",
    [
        ("One. Two. Three.", 12),
        ("One very long sentence without any punctuation at all", 14),
        ("a" * 5000, 600),
        ("Hello!\n\nQuoted ‘speech.’  More?\tYes!", 20),
        ("漢字🙂 é" * 100, 40),
    ],
)
def test_chunks_preserve_all_text_and_special_token_budget(text, limit):
    count = lambda value: len(value.encode("utf-8")) + 2
    chunks = split_text(text, count, limit)
    assert "".join(chunks) == text
    assert all(chunk and count(chunk) <= limit for chunk in chunks)


def test_prefers_sentence_then_word_boundaries():
    count = lambda text: len(text) + 2
    assert split_text("One. Two. Three.", count, 12) == ["One. Two. ", "Three."]
    assert split_text("one two three", count, 10) == ["one two ", "three"]
    assert split_text("abcdefghijk", count, 6) == ["abcd", "efgh", "ijk"]


def test_impossible_token_budget_fails_without_looping():
    with pytest.raises(ValueError, match="single text character"):
        split_text("abc", lambda text: len(text) + 2, 2)


@pytest.mark.parametrize("failure", [None, "generation", "storage"])
def test_complete_chunked_request_holds_one_lease_until_storage(tmp_path, failure):
    from dataclasses import replace

    from test_model_loader import definition

    from app.audio.service import AudioService
    from app.metrics.collector import MetricsCollector
    from app.models.loader import ModelLoader
    from app.models.registry import ModelRegistry
    from app.models.resources import ResourceManager
    from app.models.scheduler import EngineScheduler
    from app.schemas.synthesis import SynthesisRequest
    from app.synthesis.service import SynthesisService
    from app.text_processing.service import TextProcessingService

    engine = fake_engine()
    model = replace(definition("speecht5-pretrained", tmp_path / "model"), voices=("cmu-slt",))
    resources = ResourceManager(
        profiles={model.key: 100},
        cpu_units=1,
        cpu_threads=1,
        memory_reader=lambda: 10_000,
        memory_headroom_mb=0,
    )
    factory = Mock(return_value=engine)
    metrics = MetricsCollector(memory_reader=lambda: 100)
    scheduler = EngineScheduler(ModelLoader(engine_factory=factory), resources, metrics)
    waveform = engine._model.generate_speech.return_value
    calls = []

    def generate(*args, **kwargs):
        assert resources.reserved_cpu_units == 1
        assert resources.reserved_memory_mb == 100
        assert scheduler._records[model.key].active_users == 1
        calls.append(args[0].shape[-1])
        if failure == "generation" and len(calls) == 2:
            raise RuntimeError("chunk failed")
        return waveform

    engine._model.generate_speech.side_effect = generate

    class Audio(AudioService):
        def create_artifact(self, *args, **kwargs):
            assert len(calls) == 9
            assert resources.reserved_cpu_units == 1
            assert scheduler._records[model.key].active_users == 1
            if failure == "storage":
                raise RuntimeError("storage failed")
            return super().create_artifact(*args, **kwargs)

    audio = Audio(tmp_path / "audio")
    service = SynthesisService(
        ModelRegistry([model]),
        scheduler,
        audio,
        metrics,
        TextProcessingService(),
    )
    request = SynthesisRequest(
        text="a" * 5000,
        model_id=model.key,
        voice_id="cmu-slt",
        sanitize_text=False,
        normalize_text=False,
    )
    if failure:
        with pytest.raises((InferenceError, RuntimeError), match="failed"):
            service._synthesize_sync(request)
        assert not list(audio.output_dir.iterdir())
    else:
        result = service._synthesize_sync(request)
        assert result.metrics.audio_duration_ms == 90
        assert result.normalized_text == request.text
        assert len(list(audio.output_dir.iterdir())) == 1
    factory.assert_called_once()
    assert resources.reserved_cpu_units == resources.reserved_memory_mb == 0
    assert scheduler._records[model.key].active_users == 0
    assert scheduler._records[model.key].engine is engine
