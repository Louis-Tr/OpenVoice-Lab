from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from training.full_training.checkpoint import verify_checkpoint

VARIANTS = ("v1-baseline", "v2-term-balance", "v3-replay")
EXPECTED_STEPS = (125, 250, 375, 500, 625, 750, 875, 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_final(output: Path) -> dict[str, Any]:
    marker = json.loads((output / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    manifest_path = output / "run_artifact_manifest.json"
    if marker["artifact_manifest_sha256"] != _sha256(manifest_path):
        raise RuntimeError(f"final manifest digest failed: {output.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        file = output / entry["path"]
        if not file.is_file() or file.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(
                f"missing or truncated artifact: {output.name}/{entry['path']}"
            )
        if _sha256(file) != entry["sha256"]:
            raise RuntimeError(f"artifact digest failed: {output.name}/{entry['path']}")
    return {
        "verified": True,
        "file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(manifest_path),
    }


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit(root: Path) -> dict[str, Any]:
    state = json.loads((root / "orchestrator-state.json").read_text(encoding="utf-8"))
    incidents_path = root / "controller-incidents.json"
    incidents = (
        json.loads(incidents_path.read_text(encoding="utf-8"))
        if incidents_path.is_file()
        else []
    )
    variants: dict[str, Any] = {}
    estimated_cost = 0.0
    gpu_hours = 0.0
    for variant in VARIANTS:
        output = root / variant
        checkpoints = []
        for step in EXPECTED_STEPS:
            checkpoint = output / "checkpoints" / f"checkpoint-{step}"
            valid, reason = verify_checkpoint(checkpoint)
            if not valid:
                raise RuntimeError(f"{variant} checkpoint-{step} failed: {reason}")
            checkpoints.append(
                {
                    "step": step,
                    "path": str(checkpoint),
                    "manifest_sha256": _sha256(checkpoint / "checkpoint_manifest.json"),
                    "verified": True,
                }
            )
        final = _verify_final(output)
        result = json.loads((output / "run_result.json").read_text(encoding="utf-8"))
        training = json.loads(
            (output / "training_metadata.json").read_text(encoding="utf-8")
        )
        state_value = state["variants"][variant]
        created = _timestamp(state_value["created_utc"])
        terminated = _timestamp(state_value["pod_terminated_utc"])
        hours = (terminated - created).total_seconds() / 3600
        cost = hours * float(state_value["cost_per_hour"])
        gpu_hours += hours
        estimated_cost += cost
        evaluation = result["evaluation"]
        variants[variant] = {
            "status": result["status"],
            "pod_id": state_value["pod_id"],
            "pod_created_utc": state_value["created_utc"],
            "pod_terminated_utc": state_value["pod_terminated_utc"],
            "pod_terminated": state_value["pod_terminated"],
            "gpu_hours": hours,
            "estimated_cost_usd": cost,
            "global_step": training["global_step"],
            "best_step": int(training["best_model_checkpoint"].rsplit("-", 1)[1]),
            "best_eval_loss": training["best_metric"],
            "stopped_early": training["stopped_early"],
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "final_artifacts": final,
            "selected_model": str(output / "selected-model"),
            "evaluation": {
                "case_count": evaluation["case_count"],
                "failure_count": evaluation["failure_count"],
                "domain_term_accuracy": evaluation["domain_term_accuracy"]["accuracy"],
                "domain_terms_correct": evaluation["domain_term_accuracy"]["correct"],
                "domain_terms_total": evaluation["domain_term_accuracy"]["total"],
                "word_error_rate": evaluation["wer"],
                "average_inference_ms": evaluation["average_inference_ms"],
                "average_rtf": evaluation["average_rtf"],
                "peak_gpu_memory_mb": evaluation["peak_gpu_memory_mb"],
                "synthesis_verification": evaluation["synthesis_verification"],
            },
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": state["run_id"],
        "stability_gate_decision": state["stability_gate_decision"],
        "checkpoint_count": sum(
            value["checkpoint_count"] for value in variants.values()
        ),
        "all_checkpoint_hashes_verified": True,
        "all_final_artifact_hashes_verified": True,
        "all_pods_terminated": all(
            value["pod_terminated"] for value in variants.values()
        ),
        "total_gpu_hours": gpu_hours,
        "estimated_total_cost_usd": estimated_cost,
        "training_resumptions": [],
        "controller_incidents": incidents,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Stage 11 local training evidence"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/stage11/full-training")
    )
    args = parser.parse_args()
    report = audit(args.root.resolve())
    output = args.root.resolve() / "final-audit.json"
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
