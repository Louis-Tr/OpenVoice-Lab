from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.runpod_agent.core import (
    collect_status,
    create_pod,
    download_checkpoints,
    download_final_artifacts,
    read_json,
    start_training,
    terminate_pod,
    watch_run,
)


def _repository(value: str) -> Path:
    path = Path(value).resolve()
    if not (path / ".git").exists():
        raise argparse.ArgumentTypeError(f"not a Git repository root: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, run, observe, and recover RunPod SpeechT5 experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-pod", help="Create one secure RTX 4090 pod.")
    create.add_argument("--repository-root", type=_repository, default=Path.cwd())
    create.add_argument("--run-id", required=True)
    create.add_argument(
        "--approach",
        required=True,
        choices=(
            "v1a-conservative-full",
            "v1b-lora",
            "v1c-gradual-unfreeze",
            "v1d-reduction-factor-1",
        ),
    )
    create.add_argument("--document-root", type=Path)

    start = sub.add_parser(
        "start-training", help="Upload data/code and start or reattach training."
    )
    start.add_argument("--run-document", type=Path, required=True)
    start.add_argument("--pod-id", required=True)
    start.add_argument("--data-root", type=Path, default=Path.cwd())

    status = sub.add_parser("status", help="Write and print current provider/remote state.")
    status.add_argument("--run-document", type=Path, required=True)

    checkpoints = sub.add_parser(
        "download-checkpoints", help="Atomically download all completed checkpoints."
    )
    checkpoints.add_argument("--run-document", type=Path, required=True)

    final = sub.add_parser("download-final", help="Download final logs/models/evaluation.")
    final.add_argument("--run-document", type=Path, required=True)

    watch = sub.add_parser(
        "watch", help="Monitor, download checkpoints, and collect final artifacts."
    )
    watch.add_argument("--run-document", type=Path, required=True)
    watch.add_argument("--poll-seconds", type=int, default=20)
    watch.add_argument("--maximum-polls", type=int)
    watch.add_argument("--terminate-on-complete", action="store_true")

    terminate = sub.add_parser(
        "terminate", help="Terminate only after artifacts verify unless forced."
    )
    terminate.add_argument("--run-document", type=Path, required=True)
    terminate.add_argument("--force", action="store_true")
    return parser


def main(default_command: str | None = None) -> None:
    parser = _parser()
    import sys

    arguments = sys.argv[1:]
    if default_command is not None:
        arguments = [default_command, *arguments]
    args = parser.parse_args(arguments)

    if args.command == "create-pod":
        run_path = create_pod(
            args.repository_root,
            run_id=args.run_id,
            approach=args.approach,
            document_root=args.document_root,
        )
        value = read_json(run_path)
        result = {
            "run_document": str(run_path),
            "pod_document": value["pod_document"],
            "pod_id": value["pod_id"],
            "status": value["status"],
        }
    elif args.command == "start-training":
        result = start_training(
            args.run_document, pod_id=args.pod_id, data_root=args.data_root
        )
    elif args.command == "status":
        result = collect_status(args.run_document)
    elif args.command == "download-checkpoints":
        result = download_checkpoints(args.run_document)
    elif args.command == "download-final":
        result = download_final_artifacts(args.run_document)
    elif args.command == "watch":
        if args.poll_seconds < 5:
            parser.error("--poll-seconds must be at least 5")
        result = watch_run(
            args.run_document,
            poll_seconds=args.poll_seconds,
            terminate_on_complete=args.terminate_on_complete,
            maximum_polls=args.maximum_polls,
        )
    elif args.command == "terminate":
        result = terminate_pod(args.run_document, force=args.force)
    else:  # pragma: no cover
        parser.error(f"unsupported command: {args.command}")
        return
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
