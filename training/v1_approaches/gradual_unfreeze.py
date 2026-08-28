from training.v1_approaches.common import run_profile


if __name__ == "__main__":
    run_profile(
        "training/config/v1c_gradual_unfreeze.yaml",
        "v1c-gradual-unfreeze",
    )
