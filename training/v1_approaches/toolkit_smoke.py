"""Real, bounded full-model run used to validate the reusable agent toolkit."""

from training.v1_approaches.common import run_profile


if __name__ == "__main__":
    run_profile(
        "training/config/v1_toolkit_smoke.yaml",
        "v1-toolkit-smoke",
    )
