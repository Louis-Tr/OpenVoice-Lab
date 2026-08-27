"""Exercise every Stage 12 model with real CPU synthesis and pinned ASR."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from app.experiments.common import atomic_json, sha256_file
from app.experiments.model_registry import ExperimentModelRegistry
from app.experiments.scorer import score_terms, word_error_rate
from app.inference.speecht5_cpu import SpeechT5CpuRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--text", default="The patient has arm pain.")
    parser.add_argument("--term", action="append", default=["arm"])
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    stage11 = root / "artifacts" / "stage11" / "full-training"
    stage12 = root / "artifacts" / "stage12"
    cache = stage12 / "model-cache"
    output = stage12 / "verification"
    output.mkdir(parents=True, exist_ok=True)
    registry = ExperimentModelRegistry.from_artifacts(
        stage11,
        cache / "pretrained-speecht5",
        "30fcde30f19b87502b8435427b5f5068e401d5f6",
    )
    runtime = SpeechT5CpuRuntime(
        vocoder_root=cache / "vocoder",
        asr_root=cache / "asr",
        speaker_embedding_path=stage12 / "serving-profile" / "speaker-embedding.npy",
        maximum_cached_models=2,
        cpu_threads=args.cpu_threads,
    )
    runtime.prepare()
    results = []
    for summary in registry.list():
        if not summary.available or not summary.model_sha256:
            raise RuntimeError(f"Model is unavailable: {summary.id}")
        audio_path = output / f"{summary.id}.wav"
        measured = runtime.synthesize(registry.get(summary.id), args.text, audio_path)
        transcript, asr_ms = runtime.transcribe(audio_path)
        terms = score_terms(args.term, transcript)
        results.append(
            {
                "model_id": summary.id,
                "audio": audio_path.name,
                "audio_sha256": sha256_file(audio_path),
                "transcript": transcript,
                "term_score": terms.model_dump(mode="json"),
                "word_error_rate": word_error_rate(args.text, transcript),
                "model_load_ms": measured.model_load_ms,
                "inference_ms": measured.inference_ms,
                "audio_duration_ms": measured.audio_duration_ms,
                "real_time_factor": measured.real_time_factor,
                "process_memory_mb": measured.process_memory_mb,
                "asr_ms": asr_ms,
                "warm": measured.warm,
            }
        )
        print(f"{summary.id}: synthesis and ASR passed", flush=True)
    result_path = output / "runtime-verification.json"
    atomic_json(
        result_path,
        {
            "schema_version": 1,
            "completed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "runtime": "CPU",
            "text": args.text,
            "target_terms": args.term,
            "results": results,
        },
    )
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
