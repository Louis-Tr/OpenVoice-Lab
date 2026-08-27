from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import build_preflight_report, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the locked Stage 11 full-training configuration."
    )
    parser.add_argument(
        "--config",
        default="training/config/full_training.yaml",
        help="Path to the full-training YAML file.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/stage11/full-training/preflight.json",
        help="Path for the machine-readable report.",
    )
    parser.add_argument(
        "--require-launch-ready",
        action="store_true",
        help="Fail when the matching final-batch stability probe has not passed.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = build_preflight_report(config)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["configuration_valid"]:
        return 1
    if args.require_launch_ready and not report["launch_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
