from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from speechbrain.inference.speaker import EncoderClassifier
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    set_seed,
)

from training.smoke.safety import speech_labels_for_example


@dataclass
class SpeechT5ProbeCollator:
    processor: SpeechT5Processor
    reduction_factor: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{"input_ids": feature["input_ids"]} for feature in features]
        targets = [{"input_values": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_ids=inputs,
            labels=targets,
            return_tensors="pt",
        )
        batch["labels"] = batch["labels"].masked_fill(
            batch["decoder_attention_mask"].unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]
        if self.reduction_factor > 1:
            length = batch["labels"].shape[1]
            batch["labels"] = batch["labels"][
                :, : length - length % self.reduction_factor
            ]
        batch["speaker_embeddings"] = torch.tensor(
            [feature["speaker_embeddings"] for feature in features],
            dtype=torch.float32,
        )
        return batch


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _speaker_encoder(config: dict[str, Any], cache: Path) -> EncoderClassifier:
    from huggingface_hub import snapshot_download

    models = config["models"]
    source = snapshot_download(
        repo_id=models["speaker_encoder_id"],
        revision=models["speaker_encoder_revision"],
        cache_dir=cache,
    )
    return EncoderClassifier.from_hparams(
        source=source, run_opts={"device": "cuda"}
    )


def _speaker_embedding(
    encoder: EncoderClassifier, audio: np.ndarray
) -> np.ndarray:
    waveform = torch.from_numpy(audio).unsqueeze(0).to("cuda")
    with torch.no_grad():
        values = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
    embedding = values.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm else embedding


def _features(
    rows: list[dict[str, Any]],
    root: Path,
    processor: SpeechT5Processor,
    encoder: EncoderClassifier,
    sample_rate: int,
) -> list[dict[str, Any]]:
    features = []
    for row in rows:
        path = root / row["audio"]
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != sample_rate or audio.ndim != 1:
            raise ValueError(f"Expected mono {sample_rate} Hz audio: {path}")
        processed = processor(
            text=row["text"],
            audio_target=audio,
            sampling_rate=rate,
            return_attention_mask=False,
        )
        features.append(
            {
                "input_ids": processed["input_ids"],
                "labels": speech_labels_for_example(processed["labels"]),
                "speaker_embeddings": _speaker_embedding(encoder, audio),
            }
        )
    return features


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, list[str]]:
    invalid = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            invalid.append(name)
    return not invalid, invalid[:20]


def run(config_path: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    if bool(training["bf16"]) == bool(training["fp16"]):
        raise ValueError("Exactly one of BF16 and FP16 must be enabled")

    seed = int(config["seed"])
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    manifest = root / config["dataset"]["manifest"]
    rows = sorted(
        _jsonl(manifest),
        key=lambda row: (
            -float(row.get("duration_seconds") or 0.0),
            str(row["sample_id"]),
        ),
    )[: int(config["dataset"]["longest_examples"])]
    batch_size = int(training["batch_size"])
    if len(rows) != batch_size:
        raise ValueError(
            f"Probe requires exactly {batch_size} longest examples, found {len(rows)}"
        )

    for row in rows:
        audio_path = root / row["audio"]
        if _sha256(audio_path) != row["audio_sha256"]:
            raise ValueError(f"Audio checksum mismatch: {audio_path}")

    cache = root / "artifacts/stage11/stability/model-cache"
    models = config["models"]
    processor = SpeechT5Processor.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model = SpeechT5ForTextToSpeech.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model.config.use_cache = False
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
    model.to("cuda").train()
    encoder = _speaker_encoder(config, cache)
    features = _features(
        rows,
        root,
        processor,
        encoder,
        int(config["dataset"]["sample_rate"]),
    )
    del encoder
    torch.cuda.empty_cache()

    collator = SpeechT5ProbeCollator(processor, model.config.reduction_factor)
    batch = {
        key: value.to("cuda")
        for key, value in collator(features).items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"])
    )
    accumulation = int(training["gradient_accumulation_steps"])
    optimizer_steps = int(training["optimizer_steps"])
    losses: list[float] = []
    gradient_norms: list[float] = []
    step_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for optimizer_step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        for _ in range(accumulation):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite BF16 loss at optimizer step {optimizer_step + 1}"
                )
            losses.append(float(loss.detach().float().cpu()) * accumulation)
            loss.backward()
        gradients_finite, invalid_parameters = _finite_gradients(model)
        if not gradients_finite:
            raise FloatingPointError(
                f"Non-finite gradients: {invalid_parameters}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["max_grad_norm"])
        )
        if not math.isfinite(float(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm}")
        gradient_norms.append(float(gradient_norm))
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)

    elapsed = time.perf_counter() - started
    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    maximum_mb = float(training["maximum_peak_allocated_memory_mb"])
    passed = peak_allocated_mb <= maximum_mb
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "failure_reason": None
        if passed
        else (
            f"Peak allocated memory {peak_allocated_mb:.2f} MB exceeds "
            f"configured limit {maximum_mb:.2f} MB"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory
            / 1024**2,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "configuration": training,
        "manifest_sha256": _sha256(manifest),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "duration_seconds": row["duration_seconds"],
                "audio_sha256": row["audio_sha256"],
            }
            for row in rows
        ],
        "measurements": {
            "losses": losses,
            "gradient_norms_before_clipping": gradient_norms,
            "optimizer_step_seconds": step_seconds,
            "elapsed_seconds": elapsed,
            "peak_allocated_memory_mb": peak_allocated_mb,
            "peak_reserved_memory_mb": peak_reserved_mb,
            "memory_limit_mb": maximum_mb,
            "finite_losses": all(math.isfinite(value) for value in losses),
            "finite_gradients": True,
        },
    }
    output = root / config["output"]
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if not passed:
        raise RuntimeError(result["failure_reason"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeechT5 RTX 4090 stability probe")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/config/stability_probe.yaml"),
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
