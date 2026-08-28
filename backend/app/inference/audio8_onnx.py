"""CPU-only Audio8 INT4 ONNX adapter.

The DualAR generation loop is adapted from Audio8's Apache-2.0 ONNX runtime
at commit 55c1260df0176f0f7ba64e53ae0828cff3f95081. OpenVoice Lab deliberately
uses Audio8's supported no-reference prompt so the public API does not expose
voice cloning or require a reference-audio registration service.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from app.inference.base import (
    AudioResult,
    InferenceError,
    TTSInferenceEngine,
    UnsupportedVoiceError,
)

_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _clean_text(text: str) -> str:
    value = "".join(
        char if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
        for char in str(text)
    )
    return re.sub(r"\s+", " ", value).strip()


class _PromptBuilder:
    def __init__(self, tokenizer_path: Path, num_codebooks: int) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._num_codebooks = int(num_codebooks)

    def _encode(self, *parts: str) -> list[int]:
        return [
            token
            for part in parts
            for token in self._tokenizer.encode(part, add_special_tokens=False).ids
        ]

    def build_unconditioned(self, text: str) -> np.ndarray:
        target = _clean_text(text)
        if not target:
            raise ValueError("text must not be empty")
        tokens = self._encode(
            "<|im_start|>system\n",
            "convert the provided text to speech",
            "<|im_end|>\n",
            "<|im_start|>user\n",
            target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        )
        values = np.zeros((self._num_codebooks + 1, len(tokens)), dtype=np.int64)
        values[0] = np.asarray(tokens, dtype=np.int64)
        return values[np.newaxis]


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    base = np.exp(sorted_values - np.max(sorted_values))
    base /= base.sum()
    cumulative = np.cumsum(base)
    remove = (cumulative > float(top_p)) | (np.arange(base.size) >= int(top_k))
    remove[0] = False
    masked = values.copy()
    masked[order[remove]] = -np.inf
    scaled = masked / max(float(temperature), 1e-5)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    noise = -np.log(np.clip(rng.random(probabilities.size), 1e-12, 1.0))
    return int(np.argmax(probabilities / noise))


class _Audio8Runtime:
    def __init__(self, model_root: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        self._ort = ort
        self._root = model_root.resolve()
        self._manifest = json.loads(
            (self._root / "runtime_manifest.json").read_text(encoding="utf-8")
        )
        precision = "int4"
        if precision not in self._manifest["available_precisions"]:
            raise ValueError("Audio8 artifact does not provide INT4 inference")
        codec_precision = self._manifest.get("default_codec_precision", "fp16")
        self._slow = self._session(self._root / f"slow_ar_{precision}.onnx", threads)
        self._fast = self._session(self._root / f"fast_ar_{precision}.onnx", threads)
        codec_models = self._manifest.get(
            "codec_models", {"fp16": "codec_decoder_fp16.onnx"}
        )
        self._decoder = self._session(self._root / codec_models[codec_precision], threads)
        self._prompt = _PromptBuilder(
            self._root / "tokenizer" / "tokenizer.json",
            int(self._manifest["num_codebooks"]),
        )
        self._slow_inputs = {item.name: item for item in self._slow.get_inputs()}
        self._fast_inputs = {item.name: item for item in self._fast.get_inputs()}

    def _session(self, path: Path, threads: int | None) -> Any:
        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if threads is not None:
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = max(1, int(threads) // 2)
        return self._ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def _empty_slow_caches(self) -> list[np.ndarray]:
        dtype = _ORT_DTYPES[self._slow_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self._manifest["n_local_heads"]),
            int(self._manifest["max_seq_len"]),
            int(self._manifest["head_dim"]),
        )
        return [
            np.zeros(shape, dtype=dtype)
            for _ in range(2 * int(self._manifest["num_layers"]))
        ]

    def _empty_fast_caches(self) -> list[np.ndarray]:
        dtype = _ORT_DTYPES[self._fast_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self._manifest["fast_n_local_heads"]),
            int(self._manifest["num_codebooks"]),
            int(self._manifest["fast_head_dim"]),
        )
        return [
            np.zeros(shape, dtype=dtype)
            for _ in range(2 * int(self._manifest["num_fast_layers"]))
        ]

    @staticmethod
    def _update_caches(
        caches: list[np.ndarray], positions: np.ndarray, deltas: list[np.ndarray]
    ) -> None:
        for index, delta in enumerate(deltas):
            caches[index][:, :, positions, :] = delta

    def _slow_step(
        self, codes: np.ndarray, positions: np.ndarray, caches: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        feeds = {"codes": codes.astype(np.int64), "input_pos": positions.astype(np.int64)}
        for index in range(int(self._manifest["num_layers"])):
            feeds[f"cache_key_{index}"] = caches[2 * index]
            feeds[f"cache_value_{index}"] = caches[2 * index + 1]
        outputs = self._slow.run(None, feeds)
        self._update_caches(caches, positions, outputs[2:])
        return np.asarray(outputs[0])[0, -1], np.asarray(outputs[1])[:, -1:, :]

    def _fast_step(
        self,
        hidden: np.ndarray,
        token_id: int,
        use_hidden: bool,
        position: int,
        caches: list[np.ndarray],
    ) -> np.ndarray:
        hidden_dtype = _ORT_DTYPES[self._fast_inputs["slow_hidden"].type]
        feeds = {
            "slow_hidden": np.asarray(hidden, dtype=hidden_dtype),
            "token_id": np.asarray([[token_id]], dtype=np.int64),
            "use_slow_hidden": np.asarray([use_hidden], dtype=np.bool_),
            "input_pos": np.asarray([position], dtype=np.int64),
        }
        for index in range(int(self._manifest["num_fast_layers"])):
            feeds[f"cache_key_{index}"] = caches[2 * index]
            feeds[f"cache_value_{index}"] = caches[2 * index + 1]
        outputs = self._fast.run(None, feeds)
        self._update_caches(caches, np.asarray([position]), outputs[1:])
        return np.asarray(outputs[0])[0, -1]

    def _sample_semantic(
        self,
        logits: np.ndarray,
        previous: list[int],
        temperature: float,
        top_p: float,
        top_k: int,
        rng: np.random.Generator,
    ) -> int:
        begin = int(self._manifest["semantic_begin_id"])
        end = int(self._manifest["semantic_end_id"])
        stop = int(self._manifest["im_end_id"])
        allowed_ids = np.concatenate([np.arange(begin, end + 1), np.asarray([stop])])
        values = np.asarray(logits).reshape(-1)
        allowed_logits = (
            values
            if self._manifest.get("slow_logits_layout") == "semantic_then_eos"
            else values[allowed_ids]
        )
        if allowed_logits.size != allowed_ids.size:
            raise ValueError(
                f"unexpected slow logits size: {allowed_logits.size}, "
                f"expected {allowed_ids.size}"
            )
        normal = int(
            allowed_ids[_sample(allowed_logits, temperature, top_p, top_k, rng)]
        )
        high = int(allowed_ids[_sample(allowed_logits, 1.0, 0.9, top_k, rng)])
        if begin <= normal <= end and normal in previous:
            return high
        return normal

    def iter_codes(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
    ) -> Iterator[np.ndarray]:
        prompt = self._prompt.build_unconditioned(text)
        prompt_len = int(prompt.shape[2])
        max_seq_len = int(self._manifest["max_seq_len"])
        if prompt_len >= max_seq_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds max sequence length {max_seq_len}"
            )
        max_new_tokens = min(int(max_new_tokens), max_seq_len - prompt_len)
        rng = np.random.default_rng(int(seed))
        slow_caches = self._empty_slow_caches()
        logits, hidden = self._slow_step(
            prompt, np.arange(prompt_len, dtype=np.int64), slow_caches
        )
        previous: list[int] = []
        begin = int(self._manifest["semantic_begin_id"])
        stop = int(self._manifest["im_end_id"])
        codebook_size = int(self._manifest["codebook_size"])

        for step in range(max_new_tokens):
            semantic = self._sample_semantic(
                logits, previous, temperature, top_p, top_k, rng
            )
            if semantic == stop:
                return
            previous = (previous + [semantic])[-10:]
            fast_caches = self._empty_fast_caches()
            self._fast_step(hidden, 0, True, 0, fast_caches)
            token = min(max(semantic - begin, 0), codebook_size - 1)
            codebooks = [token]
            for fast_position in range(1, int(self._manifest["num_codebooks"])):
                fast_logits = self._fast_step(
                    hidden, token, False, fast_position, fast_caches
                )
                token = _sample(fast_logits, temperature, top_p, top_k, rng)
                codebooks.append(token)
            frame = np.asarray(codebooks, dtype=np.int64)
            yield frame
            if step + 1 >= max_new_tokens:
                return
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            logits, hidden = self._slow_step(
                column, np.asarray([prompt_len + step], dtype=np.int64), slow_caches
            )

    def synthesize(self, text: str, *, max_new_tokens: int, seed: int) -> np.ndarray:
        frames = list(
            self.iter_codes(
                text,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                seed=seed,
            )
        )
        if not frames:
            raise RuntimeError("model produced no codec frames")
        codes = np.stack(frames, axis=1)[np.newaxis]
        audio = self._decoder.run(None, {"codes": codes})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)


RuntimeFactory = Callable[[Path, int | None], Any]


class Audio8ONNXEngine(TTSInferenceEngine):
    """Serve Audio8's official INT4 ONNX export on CPU."""

    def __init__(
        self,
        model_root: Path,
        *,
        voice_id: str = "unconditioned",
        threads: int | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        try:
            self._runtime = (runtime_factory or _Audio8Runtime)(model_root, threads)
        except Exception as error:
            raise InferenceError(f"Failed to initialize Audio8 INT4 ONNX: {error}") from error
        self._voices = (voice_id,)
        self._inference_lock = threading.Lock()

    @property
    def voices(self) -> tuple[str, ...]:
        return self._voices

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> AudioResult:
        del language
        if voice not in self._voices:
            raise UnsupportedVoiceError(f"Voice '{voice}' is not available.")
        if speed != 1.0:
            raise InferenceError("Audio8 INT4 does not expose speed control.")
        max_new_tokens = min(1024, max(64, len(text) * 8))
        try:
            with self._inference_lock:
                samples = self._runtime.synthesize(
                    text, max_new_tokens=max_new_tokens, seed=42
                )
        except Exception as error:
            raise InferenceError(f"Audio8 synthesis failed: {error}") from error
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise InferenceError("Audio8 returned an invalid audio buffer.")
        return AudioResult(samples=values, sample_rate_hz=44_100)
