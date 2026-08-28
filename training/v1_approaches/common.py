from __future__ import annotations

import argparse
from pathlib import Path


def run_profile(default_config: str, default_variant: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the reusable {default_variant} SpeechT5 training loop."
    )
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--variant", default=default_variant)
    args = parser.parse_args()

    # Import lazily so --help and orchestration preflight do not require CUDA packages.
    from training.full_training.run import run

    run(args.config, args.variant)
