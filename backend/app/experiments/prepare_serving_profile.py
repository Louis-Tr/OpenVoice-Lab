"""Create the deterministic speaker profile used by all Stage 12 comparisons."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from app.experiments.common import atomic_json, sha256_file


def prepare_speaker_profile(
    *,
    repo_root: Path,
    manifest_path: Path,
    encoder_root: Path,
    output_root: Path,
    encoder_revision: str,
) -> Path:
    """Reproduce the Stage 11 fixed validation speaker embedding on CPU."""
    try:
        import soundfile as sf
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as error:
        raise RuntimeError(
            "Speaker-profile dependencies are unavailable. Install backend[experiment]."
        ) from error

    first_line = manifest_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(first_line)
    audio_path = (repo_root / row["audio"]).resolve()
    if sha256_file(audio_path) != row["audio_sha256"]:
        raise RuntimeError("The speaker reference audio does not match the locked manifest.")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != 16_000 or audio.ndim != 1:
        raise RuntimeError("The speaker reference must be mono 16 kHz audio.")

    # SpeechBrain 1.0 attempts to symlink its optional custom.py even when source and
    # savedir are the same. Temporary local files avoid privileged Windows symlinks
    # and are removed again so the pinned download remains an immutable cache.
    custom_module = encoder_root / "custom.py"
    label_checkpoint = encoder_root / "label_encoder.ckpt"
    created_custom_module = not custom_module.exists()
    created_label_checkpoint = not label_checkpoint.exists()
    if created_custom_module:
        custom_module.touch()
    if created_label_checkpoint:
        shutil.copyfile(encoder_root / "label_encoder.txt", label_checkpoint)
    try:
        encoder = EncoderClassifier.from_hparams(
            source=str(encoder_root.resolve()),
            savedir=str(encoder_root.resolve()),
            overrides={"pretrained_path": str(encoder_root.resolve())},
            run_opts={"device": "cpu"},
        )
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        encoder.eval()
        with torch.inference_mode():
            embedding = encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze()
    finally:
        if created_custom_module:
            custom_module.unlink(missing_ok=True)
        if created_label_checkpoint:
            label_checkpoint.unlink(missing_ok=True)
    values = embedding.detach().cpu().numpy().astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm:
        values /= norm
    if not values.size or not np.isfinite(values).all():
        raise RuntimeError("The generated speaker embedding is invalid.")

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / ".speaker-embedding.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    final = output_root / "speaker-embedding.npy"
    temporary.replace(final)
    profile = {
        "schema_version": 1,
        "reference_sample_id": row["sample_id"],
        "reference_audio_sha256": row["audio_sha256"],
        "speaker_encoder_revision": encoder_revision,
        "embedding_dimensions": int(values.size),
        "embedding_sha256": sha256_file(final),
        "device": "cpu",
    }
    atomic_json(output_root / "speaker-profile.json", profile)
    atomic_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "files": [
                {
                    "path": "speaker-embedding.npy",
                    "bytes": final.stat().st_size,
                    "sha256": profile["embedding_sha256"],
                }
            ],
        },
    )
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data-processing/manifests/stage11/v1-baseline/validation.jsonl"),
    )
    parser.add_argument(
        "--encoder-root", type=Path, default=Path("artifacts/stage12/model-cache/speaker-encoder")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/stage12/serving-profile")
    )
    parser.add_argument(
        "--encoder-revision", default="56895a2df401be4150a159f3a1c653f00051d477"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else repo_root / value
    path = prepare_speaker_profile(
        repo_root=repo_root,
        manifest_path=resolve(args.manifest),
        encoder_root=resolve(args.encoder_root),
        output_root=resolve(args.output_root),
        encoder_revision=args.encoder_revision,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
