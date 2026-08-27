from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf
import torch
import yaml
from speechbrain.inference.speaker import EncoderClassifier
from torch.utils.data import Sampler
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

from training.full_training.checkpoint import (
    CHECKPOINT_MANIFEST,
    CHECKPOINT_MARKER,
    verify_checkpoint,
)
from training.full_training.config import validate_values
from training.smoke.metrics import comparison_metrics
from training.smoke.safety import (
    non_finite_training_values,
    speech_labels_for_example,
)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_manifest(output: Path) -> dict[str, Any]:
    included_roots = [
        output / "selected-model",
        output / "evaluation",
        output / "run_provenance.json",
        output / "training_metadata.json",
        output / "pip-freeze.txt",
        output / "run_result.json",
    ]
    files: list[Path] = []
    for root in included_roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(file for file in root.rglob("*") if file.is_file())
    return {
        "schema_version": 1,
        "created_utc": _utc(),
        "files": [
            {
                "path": file.relative_to(output).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": _sha256(file),
            }
            for file in sorted(files)
        ],
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:60]


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[dict[str, Any]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class DeterministicBlockShuffleSampler(Sampler[int]):
    """Shuffle complete schedule blocks while preserving every block's row order."""

    def __init__(self, size: int, block_size: int, seed: int):
        if size % block_size:
            raise ValueError(
                f"dataset size {size} is not divisible by block size {block_size}"
            )
        self.size = size
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        blocks = list(range(self.size // self.block_size))
        random.Random(self.seed + self.epoch).shuffle(blocks)
        for block in blocks:
            start = block * self.block_size
            yield from range(start, start + self.block_size)

    def __len__(self) -> int:
        return self.size


class BlockShuffleTrainer(Seq2SeqTrainer):
    def __init__(self, *args, block_size: int, sampler_seed: int, **kwargs):
        self.block_size = block_size
        self.sampler_seed = sampler_seed
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Sampler[int] | None:
        if self.train_dataset is None:
            return None
        return DeterministicBlockShuffleSampler(
            len(self.train_dataset), self.block_size, self.sampler_seed
        )


@dataclass
class SpeechT5Collator:
    processor: SpeechT5Processor
    reduction_factor: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{"input_ids": feature["input_ids"]} for feature in features]
        targets = [{"input_values": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_ids=inputs, labels=targets, return_tensors="pt"
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


class FiniteAndProgressCallback(TrainerCallback):
    def __init__(self, output: Path):
        self.output = output

    def _progress(self, state, values: dict[str, Any]) -> None:
        disk = shutil.disk_usage(self.output)
        payload = {
            "updated_utc": _utc(),
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "gpu_peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else None
            ),
            "process_memory_mb": psutil.Process().memory_info().rss / 1024**2,
            "disk_free_gb": disk.free / 1024**3,
            "latest": values,
        }
        _atomic_json(self.output / "progress.json", payload)

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = logs or {}
        invalid = non_finite_training_values(values)
        if invalid:
            raise FloatingPointError(
                f"non-finite training values at step {state.global_step}: {invalid}"
            )
        self._progress(state, values)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._progress(state, metrics or {})
        return control


class CompleteCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        output: Path,
        processor: SpeechT5Processor,
        provenance: dict[str, Any],
    ):
        self.output = output
        self.processor = processor
        self.provenance = provenance

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.processor.save_pretrained(checkpoint)
        metadata = {
            **self.provenance,
            "checkpoint_step": state.global_step,
            "progress_fraction": state.global_step / state.max_steps,
            "best_metric_at_save": state.best_metric,
            "best_checkpoint_at_save": state.best_model_checkpoint,
            "completed_utc": _utc(),
        }
        _atomic_json(checkpoint / "checkpoint_metadata.json", metadata)
        excluded = {CHECKPOINT_MANIFEST, CHECKPOINT_MARKER}
        files = sorted(
            file
            for file in checkpoint.rglob("*")
            if file.is_file() and file.name not in excluded
        )
        manifest = {
            "schema_version": 1,
            "checkpoint_step": state.global_step,
            "created_utc": _utc(),
            "files": [
                {
                    "path": file.relative_to(checkpoint).as_posix(),
                    "bytes": file.stat().st_size,
                    "sha256": _sha256(file),
                }
                for file in files
            ],
        }
        _atomic_json(checkpoint / CHECKPOINT_MANIFEST, manifest)
        _atomic_json(
            checkpoint / CHECKPOINT_MARKER,
            {
                "schema_version": 1,
                "checkpoint_step": state.global_step,
                "manifest_sha256": _sha256(checkpoint / CHECKPOINT_MANIFEST),
                "completed_utc": _utc(),
            },
        )
        _atomic_json(
            self.output / "latest_checkpoint.json",
            {
                "checkpoint": checkpoint.name,
                "step": state.global_step,
                "completed_utc": _utc(),
            },
        )
        return control


def latest_valid_checkpoint(root: Path) -> Path | None:
    candidates = sorted(
        root.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
        reverse=True,
    )
    for candidate in candidates:
        valid, _ = verify_checkpoint(candidate)
        if valid:
            return candidate
    return None


def _speaker_encoder(config: dict[str, Any], cache: Path) -> EncoderClassifier:
    from huggingface_hub import snapshot_download

    models = config["models"]
    source = snapshot_download(
        repo_id=models["speaker_encoder_id"],
        revision=models["speaker_encoder_revision"],
        cache_dir=cache,
    )
    return EncoderClassifier.from_hparams(source=source, run_opts={"device": "cuda"})


def _speaker_embedding(encoder: EncoderClassifier, path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != 16000 or audio.ndim != 1:
        raise ValueError(f"expected mono 16 kHz audio: {path}")
    waveform = torch.from_numpy(audio).unsqueeze(0).to("cuda")
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
    embedding = embedding.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm else embedding


def _prepare_features(
    rows_by_split: dict[str, list[dict[str, Any]]],
    root: Path,
    processor: SpeechT5Processor,
    encoder: EncoderClassifier,
    output: Path,
) -> tuple[dict[str, list[dict[str, Any]]], np.ndarray]:
    cache: dict[str, dict[str, Any]] = {}
    prepared: dict[str, list[dict[str, Any]]] = {}
    total_unique = len(
        {row["audio_sha256"] for rows in rows_by_split.values() for row in rows}
    )
    completed = 0
    for split, rows in rows_by_split.items():
        split_features = []
        for row in rows:
            key = row["audio_sha256"]
            feature = cache.get(key)
            if feature is None:
                audio_path = root / row["audio"]
                audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
                processed = processor(
                    text=row["text"],
                    audio_target=audio,
                    sampling_rate=rate,
                    return_attention_mask=False,
                )
                feature = {
                    "input_ids": processed["input_ids"],
                    "labels": speech_labels_for_example(processed["labels"]),
                    "speaker_embeddings": _speaker_embedding(encoder, audio_path),
                }
                cache[key] = feature
                completed += 1
                if completed == 1 or completed % 50 == 0 or completed == total_unique:
                    _atomic_json(
                        output / "feature_progress.json",
                        {
                            "updated_utc": _utc(),
                            "unique_prepared": completed,
                            "unique_total": total_unique,
                            "split": split,
                        },
                    )
                    print(f"PREPARED_UNIQUE {completed}/{total_unique}", flush=True)
            split_features.append(feature)
        prepared[split] = split_features
    fixed = cache[rows_by_split["validation"][0]["audio_sha256"]]["speaker_embeddings"]
    return prepared, fixed


def _training_arguments(config: dict[str, Any], checkpoint_root: Path):
    training = config["training"]
    stopping = config["early_stopping"]
    checkpoints = config["checkpoints"]
    return Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_root),
        per_device_train_batch_size=int(training["physical_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=int(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        max_steps=int(training["max_steps"]),
        fp16=bool(training["fp16"]),
        bf16=bool(training["bf16"]),
        max_grad_norm=float(training["max_grad_norm"]),
        logging_nan_inf_filter=False,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=int(training["evaluation_steps"]),
        save_steps=int(checkpoints["recovery_interval_steps"]),
        logging_steps=int(training["logging_steps"]),
        load_best_model_at_end=bool(stopping["load_best_model_at_end"]),
        metric_for_best_model=stopping["metric"],
        greater_is_better=bool(stopping["greater_is_better"]),
        save_total_limit=None,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
    )


def _train(
    config: dict[str, Any],
    variant: dict[str, Any],
    root: Path,
    output: Path,
    rows: dict[str, list[dict[str, Any]]],
    provenance: dict[str, Any],
) -> tuple[Path, dict[str, Any], np.ndarray]:
    models = config["models"]
    cache = output / "model-cache"
    processor = SpeechT5Processor.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model = SpeechT5ForTextToSpeech.from_pretrained(
        models["tts_id"], revision=models["tts_revision"], cache_dir=cache
    )
    model.config.use_cache = bool(config["training"]["model_use_cache"])
    encoder = _speaker_encoder(config, cache)
    features, fixed_embedding = _prepare_features(
        rows, root, processor, encoder, output
    )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume = latest_valid_checkpoint(checkpoint_root)
    callbacks: list[TrainerCallback] = [
        FiniteAndProgressCallback(output),
        CompleteCheckpointCallback(output, processor, provenance),
    ]
    if config["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(
                    config["early_stopping"]["patience_evaluations"]
                ),
                early_stopping_threshold=float(config["early_stopping"]["threshold"]),
            )
        )
    trainer = BlockShuffleTrainer(
        model=model,
        args=_training_arguments(config, checkpoint_root),
        train_dataset=ManifestDataset(features["train"]),
        eval_dataset=ManifestDataset(features["validation"]),
        data_collator=SpeechT5Collator(processor, model.config.reduction_factor),
        tokenizer=processor,
        callbacks=callbacks,
        block_size=int(config["dataset"]["schedule_block_size"]),
        sampler_seed=int(config["seed"]),
    )
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    elapsed = time.perf_counter() - started
    selected = output / "selected-model"
    trainer.save_model(selected)
    processor.save_pretrained(selected)
    metadata = {
        "schema_version": 1,
        "variant": variant["id"],
        "completed_utc": _utc(),
        "elapsed_seconds": elapsed,
        "resumed_from": str(resume) if resume else None,
        "train_metrics": result.metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "configured_max_steps": int(config["training"]["max_steps"]),
        "stopped_early": trainer.state.global_step
        < int(config["training"]["max_steps"]),
        "log_history": trainer.state.log_history,
        "selected_model": str(selected),
    }
    _atomic_json(output / "training_metadata.json", metadata)
    del trainer, model, features
    gc.collect()
    torch.cuda.empty_cache()
    return selected, metadata, fixed_embedding


def _evaluate(
    config: dict[str, Any],
    model_path: Path,
    rows: list[dict[str, Any]],
    embedding: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    models = config["models"]
    cache = output / "model-cache"
    processor = SpeechT5Processor.from_pretrained(model_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to("cuda").eval()
    vocoder = (
        SpeechT5HifiGan.from_pretrained(
            models["vocoder_id"], revision=models["vocoder_revision"], cache_dir=cache
        )
        .to("cuda")
        .eval()
    )
    speaker = torch.tensor(embedding, dtype=torch.float32, device="cuda").unsqueeze(0)
    audio_root = output / "evaluation" / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows, start=1):
        result = {
            "sample_id": row["sample_id"],
            "reference": row["text"],
            "medical_terms": row.get("medical_terms", []),
        }
        try:
            input_ids = processor(text=row["text"], return_tensors="pt")[
                "input_ids"
            ].to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                waveform = model.generate_speech(input_ids, speaker, vocoder=vocoder)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            values = waveform.detach().float().cpu().numpy()
            path = audio_root / f"{index:04d}-{_slug(row['sample_id'])}.wav"
            sf.write(path, values, 16000, subtype="PCM_16")
            duration = len(values) / 16000
            result.update(
                audio=path.relative_to(output).as_posix(),
                audio_sha256=_sha256(path),
                inference_ms=elapsed * 1000,
                audio_duration_seconds=duration,
                rtf=elapsed / duration,
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
                process_memory_mb=psutil.Process().memory_info().rss / 1024**2,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - evaluation records per-case failures
            result.update(error=f"TTS {type(exc).__name__}: {exc}", transcript="")
        results.append(result)
        if index == 1 or index % 10 == 0 or index == len(rows):
            _atomic_json(
                output / "evaluation" / "progress.json",
                {
                    "stage": "synthesis",
                    "completed": index,
                    "total": len(rows),
                    "updated_utc": _utc(),
                },
            )
            print(f"EVALUATION_SYNTHESIS {index}/{len(rows)}", flush=True)
    del model, vocoder, processor, speaker
    gc.collect()
    torch.cuda.empty_cache()

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=models["asr_id"],
        revision=models["asr_revision"],
        device=0,
        torch_dtype=torch.float16,
        model_kwargs={"cache_dir": str(cache)},
    )
    for index, row in enumerate(results, start=1):
        if not row.get("error"):
            try:
                audio, rate = sf.read(
                    output / row["audio"], dtype="float32", always_2d=False
                )
                row["transcript"] = recognizer({"array": audio, "sampling_rate": rate})[
                    "text"
                ].strip()
            except Exception as exc:  # noqa: BLE001 - evaluation records per-case failures
                row.update(error=f"ASR {type(exc).__name__}: {exc}", transcript="")
        if index == 1 or index % 10 == 0 or index == len(results):
            _atomic_json(
                output / "evaluation" / "progress.json",
                {
                    "stage": "asr",
                    "completed": index,
                    "total": len(results),
                    "updated_utc": _utc(),
                },
            )
            print(f"EVALUATION_ASR {index}/{len(results)}", flush=True)
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()

    evaluation_root = output / "evaluation"
    with (evaluation_root / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = [
        "sample_id",
        "reference",
        "transcript",
        "audio",
        "inference_ms",
        "audio_duration_seconds",
        "rtf",
        "peak_gpu_memory_mb",
        "process_memory_mb",
        "error",
    ]
    with (evaluation_root / "raw_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summary = comparison_metrics(results)
    summary["completed_utc"] = _utc()
    summary["shared_test_cases"] = len(rows)
    summary["synthesis_verification"] = next(
        (
            {
                "status": "passed",
                "sample_id": row["sample_id"],
                "audio": row["audio"],
                "audio_sha256": row["audio_sha256"],
            }
            for row in results
            if not row.get("error")
        ),
        {"status": "failed", "reason": "no successful synthesized test case"},
    )
    _atomic_json(evaluation_root / "summary.json", summary)
    return summary


def _load_and_validate_inputs(
    root: Path, config: dict[str, Any], variant_id: str
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    errors = validate_values(config)
    if errors:
        raise ValueError("invalid training configuration: " + "; ".join(errors))
    if config["stability_gate"].get("override") != "user_authorized_skip":
        raise ValueError(
            "full training requires the recorded user-authorized stability override"
        )
    variants = {value["id"]: value for value in config["variants"]}
    if variant_id not in variants:
        raise ValueError(f"unknown variant: {variant_id}")
    variant = variants[variant_id]
    lock_path = root / config["dataset"]["lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "passed":
        raise ValueError("dataset lock is not passed")
    manifest_root = root / variant["manifest_root"]
    rows = {}
    for split in ("train", "validation", "test"):
        path = manifest_root / f"{split}.jsonl"
        digest = _sha256(path)
        expected = lock["variants"][variant_id]["manifest_sha256"][split]
        if digest != expected:
            raise ValueError(
                f"{variant_id} {split} manifest does not match dataset lock"
            )
        rows[split] = _jsonl(path)
        expected_rows = int(config["dataset"]["expected_rows"][split])
        if len(rows[split]) != expected_rows:
            raise ValueError(f"{variant_id} {split} row count changed")
    block_size = int(config["dataset"]["schedule_block_size"])
    if len(rows["train"]) % block_size:
        raise ValueError("training schedule is not composed of complete locked blocks")
    if variant_id == "v3-replay":
        for start in range(0, len(rows["train"]), block_size):
            pools = [
                row["source_pool"] for row in rows["train"][start : start + block_size]
            ]
            if pools.count("term-balanced") != 4 or pools.count("replay") != 4:
                raise ValueError(
                    f"V3 replay mix changed in block {start // block_size}"
                )
    return variant, rows, lock


def run(config_path: Path, variant_id: str) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variant, rows, lock = _load_and_validate_inputs(root, config, variant_id)
    output = root / variant["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("full training requires a CUDA GPU")
    if config["training"]["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured BF16 is unsupported by this GPU")
    set_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    config_sha = _canonical_sha256(config)
    lock_path = root / config["dataset"]["lock"]
    provenance = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "variant": variant_id,
        "run_id": os.environ.get("OPENVOICE_RUN_ID", f"local-{variant_id}"),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "created_utc": _utc(),
        "configuration_sha256": config_sha,
        "dataset_lock_sha256": _sha256(lock_path),
        "dataset_lock_status": lock["status"],
        "manifest_sha256": lock["variants"][variant_id]["manifest_sha256"],
        "models": config["models"],
        "training": config["training"],
        "early_stopping": config["early_stopping"],
        "stability_gate": config["stability_gate"],
        "stability_gate_decision": "user_authorized_skip",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory
            / 1024**2,
        },
        "dataset_counts": {split: len(values) for split, values in rows.items()},
    }
    _atomic_json(output / "run_provenance.json", provenance)
    with (output / "pip-freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, stdout=handle
        )
    selected, training, embedding = _train(
        config, variant, root, output, rows, provenance
    )
    evaluation = _evaluate(config, selected, rows["test"], embedding, output)
    result = {
        "schema_version": 1,
        "status": "completed"
        if evaluation["failure_count"] == 0
        else "completed_with_failures",
        "variant": variant_id,
        "completed_utc": _utc(),
        "training": training,
        "evaluation": evaluation,
        "selected_model": str(selected),
    }
    _atomic_json(output / "run_result.json", result)
    manifest_path = output / "run_artifact_manifest.json"
    _atomic_json(manifest_path, _artifact_manifest(output))
    _atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "schema_version": 1,
            "status": result["status"],
            "variant": variant_id,
            "completed_utc": result["completed_utc"],
            "artifact_manifest_sha256": _sha256(manifest_path),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Stage 11 full SpeechT5 experiment"
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("training/config/full_training.yaml")
    )
    args = parser.parse_args()
    try:
        run(args.config, args.variant)
    except Exception as exc:
        variant = (
            args.variant if re.fullmatch(r"v[123]-[a-z-]+", args.variant) else "unknown"
        )
        output = Path("artifacts/stage11/full-training") / variant
        _atomic_json(
            output / "RUN_FAILED.json",
            {
                "schema_version": 1,
                "status": "failed",
                "variant": variant,
                "failed_utc": _utc(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
