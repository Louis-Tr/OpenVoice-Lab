from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import yaml
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

from training.full_training.reduction_factor import (
    assert_loaded_reduction_factor,
    assert_saved_reduction_factor,
)
from training.full_training.run import _load_training_model


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path.cwd()
    variant = config["variants"][0]
    output = root / variant["output_root"]
    cache = output / "model-cache"
    expected = int(config["training"]["reduction_factor"])
    model, metadata = _load_training_model(config, cache)
    assert_loaded_reduction_factor(model, expected, Path(config["models"]["tts_id"]))
    model = model.to("cuda").train()
    processor = SpeechT5Processor.from_pretrained(
        config["models"]["tts_id"],
        revision=config["models"]["tts_revision"],
        cache_dir=cache,
    )
    input_ids = processor(
        text="The patient was prescribed amlodipine for hypertension.",
        return_tensors="pt",
    )["input_ids"].to("cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = torch.zeros((1, 100, int(model.config.num_mel_bins)), device="cuda")
    speaker = torch.ones(
        (1, int(model.config.speaker_embedding_dim)), device="cuda"
    )
    speaker = torch.nn.functional.normalize(speaker, dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speaker_embeddings=speaker,
            return_dict=True,
        )
    if not math.isfinite(float(result.loss.detach())):
        raise FloatingPointError("reduction-factor compatibility loss is non-finite")
    if result.spectrogram.shape[-1] != int(model.config.num_mel_bins):
        raise RuntimeError("reduction-factor output mel dimension is invalid")
    result.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError(
            "reduction-factor compatibility gradients are missing or non-finite"
        )

    export = output / "compatibility-smoke" / "model"
    if export.exists():
        shutil.rmtree(export)
    model.save_pretrained(export, safe_serialization=True)
    processor.save_pretrained(export)
    assert_saved_reduction_factor(export, expected)
    del model
    torch.cuda.empty_cache()
    reloaded = SpeechT5ForTextToSpeech.from_pretrained(
        export, local_files_only=True
    )
    assert_loaded_reduction_factor(reloaded, expected, export)
    del reloaded
    shutil.rmtree(export)

    report = {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": _utc(),
        "loss": float(result.loss.detach()),
        "spectrogram_shape": list(result.spectrogram.shape),
        "finite_gradient_tensors": len(gradients),
        "reduction_factor": expected,
        "approach_runtime": metadata,
    }
    _atomic_json(output / "compatibility-smoke.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise reduction-factor-1 model shape and training compatibility."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
