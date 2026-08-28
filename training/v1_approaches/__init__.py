"""Reusable SpeechT5 V1 training approach entrypoints."""

PROFILES = {
    "v1-toolkit-smoke": (
        "training.v1_approaches.toolkit_smoke",
        "training/config/v1_toolkit_smoke.yaml",
    ),
    "v1a-conservative-full": (
        "training.v1_approaches.conservative_full",
        "training/config/v1a_conservative_full.yaml",
    ),
    "v1b-lora": (
        "training.v1_approaches.lora",
        "training/config/v1b_lora.yaml",
    ),
    "v1c-gradual-unfreeze": (
        "training.v1_approaches.gradual_unfreeze",
        "training/config/v1c_gradual_unfreeze.yaml",
    ),
    "v1d-reduction-factor-1": (
        "training.v1_approaches.reduction_factor_1",
        "training/config/v1d_reduction_factor_1.yaml",
    ),
}

COMPATIBILITY_MODULES = {
    "v1b-lora": "training.full_training.lora_compatibility",
    "v1d-reduction-factor-1": "training.full_training.reduction_factor_compatibility",
}
