from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

from training.full_training.lora import apply_lora, load_adapter, merge_adapter


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path.cwd()
    variant = config["variants"][0]
    output = root / variant["output_root"]
    cache = output / "model-cache"
    smoke_root = output / "compatibility-smoke"
    models = config["models"]
    processor = SpeechT5Processor.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    base = SpeechT5ForTextToSpeech.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    ).to("cuda")
    wrapped, metadata = apply_lora(base, config)
    wrapped.train()
    input_ids = processor(
        text="The patient was prescribed amlodipine for hypertension.",
        return_tensors="pt",
    )["input_ids"].to("cuda")
    labels = torch.zeros((1, 100, int(base.config.num_mel_bins)), device="cuda")
    speaker = torch.ones((1, int(base.config.speaker_embedding_dim)), device="cuda")
    speaker = torch.nn.functional.normalize(speaker, dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = wrapped(
            input_ids=input_ids,
            labels=labels,
            speaker_embeddings=speaker,
        ).loss
    if not math.isfinite(float(loss.detach())):
        raise FloatingPointError(f"compatibility loss is not finite: {loss}")
    loss.backward()
    gradients = [
        value.grad
        for value in wrapped.parameters()
        if value.requires_grad and value.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("LoRA compatibility gradients are missing or non-finite")
    adapter = smoke_root / "adapter"
    wrapped.save_pretrained(adapter, safe_serialization=True)
    del wrapped, base
    torch.cuda.empty_cache()
    reload_base = SpeechT5ForTextToSpeech.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    ).to("cuda")
    reloaded = load_adapter(reload_base, adapter, trainable=False).eval()
    with torch.inference_mode():
        adapter_values = reloaded(
            input_ids=input_ids,
            labels=labels,
            speaker_embeddings=speaker,
        ).spectrogram.float()
    merged = merge_adapter(reloaded).eval()
    with torch.inference_mode():
        merged_values = merged(
            input_ids=input_ids,
            labels=labels,
            speaker_embeddings=speaker,
        ).spectrogram.float()
    difference = float((adapter_values - merged_values).abs().max().item())
    if difference > 0.0001:
        raise RuntimeError(f"compatibility merge difference {difference} exceeds 0.0001")
    result = {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": _utc(),
        "loss": float(loss.detach()),
        "finite_gradient_tensors": len(gradients),
        "merge_maximum_absolute_difference": difference,
        "metadata": metadata,
    }
    _atomic_json(output / "compatibility-smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise SpeechT5 PEFT LoRA compatibility")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
