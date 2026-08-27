from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import os
import platform
import random
import re
import subprocess
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
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    TrainerCallback,
    pipeline,
    set_seed,
)

from .metrics import comparison_metrics
from .safety import (
    non_finite_training_values,
    require_successful_comparison,
    speech_labels_for_example,
    validate_training_safety,
)


SMOKE_WARNING = (
    "This is a tiny pipeline smoke trial. Its quality metrics are not scientific "
    "evidence and must not be presented as a model-improvement claim."
)


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
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:60]


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[dict[str, Any]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


@dataclass
class SpeechT5Collator:
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


class FiniteTrainingCallback(TrainerCallback):
    """Stop immediately instead of saving a numerically invalid model."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        invalid = non_finite_training_values(logs or {})
        if invalid:
            raise FloatingPointError(
                f"Non-finite training values at step {state.global_step}: {invalid}"
            )
        return control


def _speaker_encoder(
    config: dict[str, Any], cache: Path, device: str
) -> EncoderClassifier:
    from huggingface_hub import snapshot_download

    models = config["models"]
    source = snapshot_download(
        repo_id=models["speaker_encoder_id"],
        revision=models["speaker_encoder_revision"],
        cache_dir=cache,
    )
    return EncoderClassifier.from_hparams(source=source, run_opts={"device": device})


def _speaker_embedding(
    encoder: EncoderClassifier, audio_path: Path, device: str
) -> np.ndarray:
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if rate != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz audio: {audio_path}")
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
    embedding = embedding.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm else embedding


def _features(
    rows: list[dict[str, Any]],
    root: Path,
    processor: SpeechT5Processor,
    encoder: EncoderClassifier,
    device: str,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows, start=1):
        path = root / row["audio"]
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        processed = processor(
            text=row["text"],
            audio_target=audio,
            sampling_rate=rate,
            return_attention_mask=False,
        )
        output.append(
            {
                "input_ids": processed["input_ids"],
                "labels": speech_labels_for_example(processed["labels"]),
                "speaker_embeddings": _speaker_embedding(encoder, path, device),
            }
        )
        print(f"PREPARED {index}/{len(rows)}", flush=True)
    return output


def _checkpoint_manifest(path: Path, step: int, config: dict[str, Any]) -> None:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    _atomic_json(
        path / "smoke_checkpoint_manifest.json",
        {
            "warning": SMOKE_WARNING,
            "optimizer_step": step,
            "configured_max_steps": config["training"]["max_steps"],
            "progress_fraction": step / config["training"]["max_steps"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": [
                {"path": file.relative_to(path).as_posix(), "sha256": _sha256(file)}
                for file in files
            ],
        },
    )


def _train(
    config: dict[str, Any],
    root: Path,
    output: Path,
    rows: dict[str, list[dict[str, Any]]],
) -> tuple[Path, dict[str, Any], np.ndarray]:
    models = config["models"]
    train_config = config["training"]
    cache = output / "model-cache"
    processor = SpeechT5Processor.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model = SpeechT5ForTextToSpeech.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model.config.use_cache = False
    encoder = _speaker_encoder(config, cache, "cuda")
    train_features = _features(rows["train"], root, processor, encoder, "cuda")
    validation_features = _features(
        rows["validation"], root, processor, encoder, "cuda"
    )
    fixed_row = rows["validation"][0] if rows["validation"] else rows["test"][0]
    fixed_embedding = _speaker_embedding(encoder, root / fixed_row["audio"], "cuda")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint_root = output / "checkpoints"
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_root),
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_steps=train_config["warmup_steps"],
        max_steps=train_config["max_steps"],
        fp16=bool(train_config["fp16"]),
        bf16=bool(train_config["bf16"]),
        max_grad_norm=float(train_config["max_grad_norm"]),
        logging_nan_inf_filter=False,
        gradient_checkpointing=bool(train_config["gradient_checkpointing"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=train_config["evaluation_steps"],
        save_steps=train_config["checkpoint_steps"],
        logging_steps=train_config["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=4,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=ManifestDataset(train_features),
        eval_dataset=ManifestDataset(validation_features),
        data_collator=SpeechT5Collator(processor, model.config.reduction_factor),
        tokenizer=processor,
        callbacks=[
            FiniteTrainingCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=train_config["early_stopping_patience"],
                early_stopping_threshold=train_config["early_stopping_threshold"],
            ),
        ],
    )
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    final = output / "adapted-model"
    trainer.save_model(final)
    processor.save_pretrained(final)
    checkpoints = sorted(
        checkpoint_root.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    for checkpoint in checkpoints:
        processor.save_pretrained(checkpoint)
        _checkpoint_manifest(checkpoint, int(checkpoint.name.rsplit("-", 1)[1]), config)
    metadata = {
        "warning": SMOKE_WARNING,
        "elapsed_seconds": elapsed,
        "train_metrics": result.metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "configured_max_steps": train_config["max_steps"],
        "stopped_early": trainer.state.global_step < train_config["max_steps"],
        "numerical_health": {
            "status": "passed",
            "precision": "bf16" if train_config["bf16"] else "fp16",
            "max_grad_norm": train_config["max_grad_norm"],
        },
        "log_history": trainer.state.log_history,
        "checkpoints": [checkpoint.name for checkpoint in checkpoints],
    }
    _atomic_json(output / "training_metadata.json", metadata)
    del trainer, model, train_features, validation_features
    gc.collect()
    torch.cuda.empty_cache()
    return final, metadata, fixed_embedding


def _generate_candidate(
    name: str,
    model_source: str | Path,
    model_revision: str | None,
    rows: list[dict[str, Any]],
    embedding: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    models = config["models"]
    cache = output / "model-cache"
    kwargs = {"cache_dir": cache}
    if model_revision:
        kwargs["revision"] = model_revision
    processor = SpeechT5Processor.from_pretrained(model_source, **kwargs)
    model = (
        SpeechT5ForTextToSpeech.from_pretrained(model_source, **kwargs)
        .to("cuda")
        .eval()
    )
    vocoder = (
        SpeechT5HifiGan.from_pretrained(
            models["vocoder_id"], revision=models["vocoder_revision"], cache_dir=cache
        )
        .to("cuda")
        .eval()
    )
    speaker = torch.tensor(embedding, dtype=torch.float32, device="cuda").unsqueeze(0)
    audio_dir = output / "comparison" / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows, start=1):
        result = {
            "candidate": name,
            "sample_id": row["sample_id"],
            "reference": row["text"],
            "medical_terms": row.get("medical_terms", []),
        }
        try:
            inputs = processor(text=row["text"], return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                waveform = model.generate_speech(input_ids, speaker, vocoder=vocoder)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            values = waveform.detach().float().cpu().numpy()
            path = audio_dir / f"{index:02d}-{_slug(row['sample_id'])}.wav"
            sf.write(path, values, 16000, subtype="PCM_16")
            duration = len(values) / 16000
            result.update(
                {
                    "audio": path.relative_to(output).as_posix(),
                    "audio_sha256": _sha256(path),
                    "inference_ms": elapsed * 1000,
                    "audio_duration_seconds": duration,
                    "rtf": elapsed / duration,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "error": None,
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        print(f"GENERATED {name} {index}/{len(rows)}", flush=True)
    del model, vocoder, processor, speaker
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _transcribe(
    config: dict[str, Any], output: Path, rows: list[dict[str, Any]]
) -> None:
    models = config["models"]
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=models["asr_id"],
        revision=models["asr_revision"],
        device=0,
        torch_dtype=torch.float16,
        model_kwargs={"cache_dir": str(output / "model-cache")},
    )
    pending = [row for row in rows if not row.get("error")]
    for index, row in enumerate(pending, start=1):
        try:
            # Passing decoded samples avoids an unnecessary ffmpeg runtime
            # dependency for the PCM WAV files generated by this trial.
            audio, sample_rate = sf.read(
                output / row["audio"], dtype="float32", always_2d=False
            )
            row["transcript"] = recognizer(
                {"array": audio, "sampling_rate": sample_rate}
            )["text"].strip()
        except Exception as exc:
            row["error"] = f"ASR {type(exc).__name__}: {exc}"
            row["transcript"] = ""
        print(f"TRANSCRIBED {index}/{len(pending)}", flush=True)
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()


def _comparison_report(
    output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    with (comparison / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = [
        "candidate",
        "sample_id",
        "reference",
        "transcript",
        "audio",
        "inference_ms",
        "audio_duration_seconds",
        "rtf",
        "peak_gpu_memory_mb",
        "error",
    ]
    with (comparison / "raw_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        audio = html.escape(row.get("audio", ""))
        cards.append(
            f"<article><h3>{html.escape(row['candidate'])} · {html.escape(row['sample_id'])}</h3>"
            f"<p><b>Reference:</b> {html.escape(row['reference'])}</p>"
            f"<p><b>ASR:</b> {html.escape(row.get('transcript', ''))}</p>"
            f"<audio controls preload='none' src='../{audio}'></audio></article>"
        )
    (comparison / "listening_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Stage 11 smoke comparison</title>"
        "<style>body{font:15px system-ui;max-width:1000px;margin:2rem auto}article{border:1px solid #bbb;padding:1rem;margin:1rem 0}</style>"
        f"<h1>Stage 11 base/adapted smoke comparison</h1><p><strong>{html.escape(SMOKE_WARNING)}</strong></p>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>{''.join(cards)}",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("The smoke trial requires a CUDA GPU")
    validate_training_safety(
        config["training"], bf16_supported=torch.cuda.is_bf16_supported()
    )
    manifest_root = root / config["dataset"]["manifest_root"]
    rows = {
        name: _jsonl(manifest_root / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    if not rows["train"] or not rows["validation"] or not rows["test"]:
        raise ValueError(
            f"Smoke trial requires non-empty train/validation/test manifests: {manifest_root}"
        )
    evaluation_rows = rows["test"][: config["dataset"]["maximum_evaluation_cases"]]
    provenance = {
        "warning": SMOKE_WARNING,
        "trial_id": config["trial_id"],
        "seed": config["seed"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": config["models"],
        "training": config["training"],
        "dataset_counts": {name: len(value) for name, value in rows.items()},
        "manifest_sha256": {
            name: _sha256(manifest_root / f"{name}.jsonl") for name in rows
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    _atomic_json(output / "trial_provenance.json", provenance)
    with (output / "pip-freeze.txt").open("w", encoding="utf-8") as freeze_output:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            stdout=freeze_output,
        )
    adapted_path, training, embedding = _train(config, root, output, rows)
    base = _generate_candidate(
        "base",
        config["models"]["tts_id"],
        config["models"]["tts_revision"],
        evaluation_rows,
        embedding,
        config,
        output,
    )
    adapted = _generate_candidate(
        "adapted", adapted_path, None, evaluation_rows, embedding, config, output
    )
    raw = base + adapted
    _transcribe(config, output, raw)
    summary = {
        "warning": SMOKE_WARNING,
        "training": training,
        "base": comparison_metrics(base),
        "adapted": comparison_metrics(adapted),
        "interpretation": (
            "No improvement claim is made from this tiny trial. Compare the measured "
            "values only to confirm the pipeline produced, trained, generated, and evaluated artifacts."
        ),
    }
    _atomic_json(output / "comparison" / "summary.json", summary)
    _comparison_report(output, raw, summary)
    require_successful_comparison(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def reevaluate(config_path: Path) -> dict[str, Any]:
    """Repeat ASR and aggregation without retraining or regenerating audio."""
    root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    raw_path = output / "comparison" / "raw_results.jsonl"
    rows = _jsonl(raw_path)
    for row in rows:
        if str(row.get("error") or "").startswith("ASR "):
            row["error"] = None
            row["transcript"] = ""
    _transcribe(config, output, rows)
    base = [row for row in rows if row["candidate"] == "base"]
    adapted = [row for row in rows if row["candidate"] == "adapted"]
    training = json.loads(
        (output / "training_metadata.json").read_text(encoding="utf-8")
    )
    summary = {
        "warning": SMOKE_WARNING,
        "training": training,
        "base": comparison_metrics(base),
        "adapted": comparison_metrics(adapted),
        "interpretation": (
            "No improvement claim is made from this tiny trial. Compare the measured "
            "values only to confirm the pipeline produced, trained, generated, and evaluated artifacts."
        ),
    }
    _atomic_json(output / "comparison" / "summary.json", summary)
    _comparison_report(output, rows, summary)
    require_successful_comparison(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 11 SpeechT5 smoke trial")
    parser.add_argument(
        "--config", type=Path, default=Path("training/config/smoke_trial.yaml")
    )
    parser.add_argument("--reevaluate-only", action="store_true")
    args = parser.parse_args()
    if args.reevaluate_only:
        reevaluate(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
