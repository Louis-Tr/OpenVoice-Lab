from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def discover_target_modules(
    model: torch.nn.Module, target_names: list[str]
) -> list[str]:
    targets = set(target_names)
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets
    )


def parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    total = sum(value.numel() for value in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_parameter_names": trainable_names,
    }


def assert_lora_only_trainable(model: torch.nn.Module) -> dict[str, Any]:
    inventory = parameter_inventory(model)
    invalid = [
        name
        for name in inventory["trainable_parameter_names"]
        if "lora_" not in name
    ]
    if invalid:
        raise RuntimeError(
            "non-LoRA parameters remain trainable: " + ", ".join(invalid[:10])
        )
    if not inventory["trainable_parameter_names"]:
        raise RuntimeError("LoRA configuration exposed no trainable parameters")
    return inventory


def apply_lora(model: torch.nn.Module, values: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    approach = values["approach"]
    target_names = list(approach["target_modules"])
    matched = discover_target_modules(model, target_names)
    expected = int(approach["expected_target_linear_modules"])
    if len(matched) != expected:
        raise RuntimeError(
            f"SpeechT5 LoRA target inspection found {len(matched)} linear modules; "
            f"expected {expected}"
        )
    lora = LoraConfig(
        r=int(approach["rank"]),
        lora_alpha=int(approach["alpha"]),
        lora_dropout=float(approach["dropout"]),
        bias=str(approach["bias"]),
        target_modules=target_names,
    )
    wrapped = get_peft_model(model, lora)
    if hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    inventory = assert_lora_only_trainable(wrapped)
    metadata = {
        "type": "peft_lora",
        "peft_version": __import__("peft").__version__,
        "rank": int(approach["rank"]),
        "alpha": int(approach["alpha"]),
        "dropout": float(approach["dropout"]),
        "bias": str(approach["bias"]),
        "target_module_suffixes": target_names,
        "matched_target_modules": matched,
        **inventory,
    }
    return wrapped, metadata


def load_adapter(base_model: torch.nn.Module, adapter_path: Path, *, trainable: bool):
    from peft import PeftModel

    return PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=trainable,
    )


def merge_adapter(model: torch.nn.Module):
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("expected a PEFT model with merge_and_unload")
    return model.merge_and_unload(safe_merge=True)
