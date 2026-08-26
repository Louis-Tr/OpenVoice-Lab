"""Kokoro ONNX adapter placeholder."""

from app.inference.base import AudioBuffer, InferenceRequest, TTSInferenceAdapter


class KokoroOnnxAdapter(TTSInferenceAdapter):
    """Future ONNX-backed implementation of the inference abstraction."""

    async def synthesize(self, _request: InferenceRequest) -> AudioBuffer:
        """Generate audio after the Kokoro runtime is integrated."""
        raise NotImplementedError("Kokoro ONNX inference is not implemented yet.")

