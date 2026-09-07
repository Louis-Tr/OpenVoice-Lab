"""SpeechT5 token boundaries without loading model weights."""

from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.inference.base import InputTooLongError
from app.inference.speecht5 import SpeechT5InferenceEngine


@pytest.mark.parametrize("tokens", [600, 601, 643])
def test_speecht5_checks_actual_tokens_before_generation(tokens: int) -> None:
    engine = SpeechT5InferenceEngine.__new__(SpeechT5InferenceEngine)
    engine._voices = ("cmu-slt",)
    engine._processor_lock = Lock()
    engine._torch = SimpleNamespace(Generator=Mock(), inference_mode=nullcontext)
    processor = Mock(return_value={"input_ids": np.zeros((1, tokens), dtype=np.int64)})
    processor.tokenizer = SimpleNamespace(model_max_length=600)
    engine._processor = processor
    waveform = Mock()
    waveform.detach().float().cpu().numpy.return_value = np.zeros(160, dtype=np.float32)
    generate = Mock(return_value=waveform)
    engine._model = SimpleNamespace(
        config=SimpleNamespace(max_text_positions=600),
        generate_speech=generate,
    )
    engine._speaker = object()
    engine._vocoder = object()

    if tokens > 600:
        with pytest.raises(InputTooLongError, match=f"received {tokens}"):
            engine.synthesize("Text expanded during cleanup.", "cmu-slt")
        generate.assert_not_called()
    else:
        result = engine.synthesize("Text at the limit.", "cmu-slt")
        assert result.sample_rate_hz == 16_000
        assert result.samples.size == 160
        generate.assert_called_once()
