from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import STAGES, run_pipeline, with_run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible, non-destructive medical TTS dataset preparation pipeline"
    )
    parser.add_argument("--config", default="training/config/dataset.yaml")
    parser.add_argument("--start-stage", choices=STAGES, default="inventory")
    parser.add_argument("--end-stage", choices=STAGES, default="report")
    parser.add_argument(
        "--limit", type=int, help="Deterministic sample limit for smoke runs"
    )
    parser.add_argument(
        "--run-name",
        help="Isolate generated output below a named subdirectory (required with --limit)",
    )
    parser.add_argument("--asr-mode", choices=("disabled", "cache_only", "full"))
    parser.add_argument(
        "--accept-unreviewed-for-smoke",
        action="store_true",
        help=(
            "Accept manual-review flags only in an isolated limited smoke run; "
            "hard validation exclusions still apply"
        ),
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help=(
            "Explicitly accept manual-review flags without reviewer decisions for a "
            "full or limited run. Hard validation and audio-quality exclusions remain."
        ),
    )
    parser.add_argument("--list-stages", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_stages:
        print("\n".join(f"{index:02d} {name}" for index, name in enumerate(STAGES)))
        return
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.limit is not None and not args.run_name:
        parser.error("--run-name is required with --limit to protect full-run outputs")
    if args.accept_unreviewed_for_smoke and args.limit is None:
        parser.error("--accept-unreviewed-for-smoke requires --limit")
    if args.accept_unreviewed_for_smoke and not args.run_name:
        parser.error("--accept-unreviewed-for-smoke requires --run-name")
    if args.accept_unreviewed_for_smoke and args.skip_review:
        parser.error(
            "--accept-unreviewed-for-smoke and --skip-review are mutually exclusive"
        )
    config = with_run_name(load_config(args.config), args.run_name)
    result = run_pipeline(
        config,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        limit=args.limit,
        asr_mode=args.asr_mode,
        accept_unreviewed_for_smoke=args.accept_unreviewed_for_smoke,
        skip_review=args.skip_review,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
