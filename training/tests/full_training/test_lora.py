from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from training.full_training.config import load_config, validate_values

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "training" / "config" / "v1b_lora.yaml"
COMPATIBILITY_PATH = (
    REPOSITORY_ROOT / "training" / "full_training" / "lora_compatibility.py"
)


def test_lora_recipe_is_locked() -> None:
    values = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert validate_values(values) == []
    assert values["expected_variant_ids"] == ["v1b-lora"]
    assert values["variants"][0]["dataset_source_variant"] == "v1-baseline"
    assert values["approach"] == {
        "type": "peft_lora",
        "base_weights_frozen": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "expected_target_linear_modules": 96,
    }
    assert values["training"]["physical_batch_size"] == 16
    assert values["training"]["gradient_accumulation_steps"] == 2
    assert values["training"]["effective_batch_size"] == 32
    assert values["training"]["learning_rate"] == 5e-5
    assert values["training"]["max_steps"] == 250
    assert values["training"]["evaluation_steps"] == 25
    assert values["checkpoints"]["recovery_interval_steps"] == 25


def test_lora_preflight_uses_locked_v1_dataset() -> None:
    config = load_config(CONFIG_PATH)
    assert config.values["variants"][0]["manifest_root"].endswith("v1-baseline")
    assert config.values["variants"][0]["output_root"].endswith("v1b-lora")


def test_adapter_loader_uses_peft_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from training.full_training.lora import load_adapter

    calls = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, path, *, is_trainable):
            calls.append((base, path, is_trainable))
            return "loaded-adapter"

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    base = torch.nn.Linear(2, 2)
    checkpoint = Path("checkpoint-25")

    assert load_adapter(base, checkpoint, trainable=False) == "loaded-adapter"
    assert calls == [(base, str(checkpoint), False)]


def test_only_lora_parameters_may_be_trainable() -> None:
    torch = pytest.importorskip("torch")
    from training.full_training.lora import assert_lora_only_trainable

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.lora_A = torch.nn.Linear(2, 1, bias=False)
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    inventory = assert_lora_only_trainable(Tiny())
    assert inventory["trainable_parameters"] == 2
    assert inventory["trainable_parameter_names"] == ["lora_A.weight"]


def test_merge_compatibility_replays_identical_rng_state() -> None:
    source = COMPATIBILITY_PATH.read_text(encoding="utf-8")

    assert "torch.manual_seed(seed)" in source
    assert "torch.cuda.manual_seed_all(seed)" in source
    assert 'comparison_seed = int(config["seed"])' in source
    assert source.count("_reset_comparison_rng(comparison_seed)") == 2
