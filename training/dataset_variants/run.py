from __future__ import annotations

import argparse
import json

from .builder import build_all_variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit the three Stage 11 training dataset schedules"
    )
    parser.add_argument(
        "--config", default="training/config/dataset_variants.yaml"
    )
    args = parser.parse_args()
    print(json.dumps(build_all_variants(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
