from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from training.full_training.gradual_unfreeze import (
    GradualUnfreezeCallback,
    GradualUnfreezeController,
    parameter_inventory,
)


class ToySpeechT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.speecht5 = torch.nn.Module()
        self.speecht5.encoder = torch.nn.Module()
        self.speecht5.encoder.prenet = torch.nn.Linear(2, 2)
        self.speecht5.encoder.wrapped_encoder = torch.nn.Module()
        self.speecht5.encoder.wrapped_encoder.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(3)]
        )
        self.speecht5.decoder = torch.nn.Module()
        self.speecht5.decoder.prenet = torch.nn.Linear(2, 2)
        self.speecht5.decoder.wrapped_decoder = torch.nn.Module()
        self.speecht5.decoder.wrapped_decoder.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(6)]
        )
        self.speech_decoder_postnet = torch.nn.Linear(2, 2)


def _approach() -> dict:
    return {
        "transition_after_step": 50,
        "phase_one_trainable_prefixes": [
            "speecht5.encoder.prenet.",
            "speecht5.decoder.prenet.",
            "speech_decoder_postnet.",
        ],
        "phase_two_decoder_prefixes": [
            "speecht5.decoder.wrapped_decoder.layers.4.",
            "speecht5.decoder.wrapped_decoder.layers.5.",
        ],
    }


def _trainable_names(model: torch.nn.Module) -> set[str]:
    return {name for name, value in model.named_parameters() if value.requires_grad}


def test_phase_one_freezes_the_shared_backbone_and_trains_only_modal_heads() -> None:
    model = ToySpeechT5()
    controller = GradualUnfreezeController(_approach())

    assert controller.apply(model, global_step=0) == "phase_1_heads"
    names = _trainable_names(model)

    assert names
    assert all(
        name.startswith(
            (
                "speecht5.encoder.prenet.",
                "speecht5.decoder.prenet.",
                "speech_decoder_postnet.",
            )
        )
        for name in names
    )
    assert not any("wrapped_encoder" in name for name in names)
    assert not any("wrapped_decoder" in name for name in names)


def test_phase_two_unfreezes_exactly_the_top_two_decoder_blocks() -> None:
    model = ToySpeechT5()
    controller = GradualUnfreezeController(_approach())

    assert controller.apply(model, global_step=50) == "phase_2_top_decoder"
    names = _trainable_names(model)

    assert any("wrapped_decoder.layers.4" in name for name in names)
    assert any("wrapped_decoder.layers.5" in name for name in names)
    assert not any("wrapped_decoder.layers.3" in name for name in names)
    assert not any("wrapped_encoder" in name for name in names)


def test_optimizer_roles_are_stable_before_and_after_transition() -> None:
    model = ToySpeechT5()
    controller = GradualUnfreezeController(_approach())
    controller.apply(model, global_step=0)

    heads, decoder = controller.optimizer_parameters(model)

    assert heads
    assert decoder
    assert all(value.requires_grad for value in heads)
    assert all(not value.requires_grad for value in decoder)
    controller.apply(model, global_step=50)
    resumed_heads, resumed_decoder = controller.optimizer_parameters(model)
    assert [id(value) for value in heads] == [id(value) for value in resumed_heads]
    assert [id(value) for value in decoder] == [id(value) for value in resumed_decoder]
    assert all(value.requires_grad for value in resumed_decoder)


def test_callback_transitions_after_step_fifty_and_writes_checkpoint_state(
    tmp_path,
) -> None:
    model = ToySpeechT5()
    controller = GradualUnfreezeController(_approach())
    controller.apply(model, global_step=0)
    heads, decoder = controller.optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": heads, "lr": 1e-6, "initial_lr": 1e-6, "group_name": "modal_tts_heads"},
            {"params": decoder, "lr": 5e-7, "initial_lr": 5e-7, "group_name": "top_two_decoder_blocks"},
        ]
    )
    callback = GradualUnfreezeCallback(
        tmp_path,
        controller,
        head_learning_rate=1e-6,
        decoder_learning_rate=5e-7,
    )
    control = SimpleNamespace()
    args = SimpleNamespace(output_dir=str(tmp_path / "checkpoints"))
    state = SimpleNamespace(global_step=0)

    callback.on_train_begin(args, state, control, model=model, optimizer=optimizer)
    state.global_step = 49
    callback.on_step_end(args, state, control, model=model, optimizer=optimizer)
    assert callback.current_phase == "phase_1_heads"
    state.global_step = 50
    callback.on_step_end(args, state, control, model=model, optimizer=optimizer)
    assert callback.current_phase == "phase_2_top_decoder"

    checkpoint = tmp_path / "checkpoints" / "checkpoint-50"
    checkpoint.mkdir(parents=True)
    callback.on_save(args, state, control, model=model, optimizer=optimizer)
    recorded = json.loads(
        (checkpoint / "gradual_unfreeze_state.json").read_text(encoding="utf-8")
    )
    assert recorded["phase"] == "phase_2_top_decoder"
    assert [group["group_name"] for group in recorded["optimizer_groups"]] == [
        "modal_tts_heads",
        "top_two_decoder_blocks",
    ]


def test_resume_at_step_fifty_restores_phase_two_without_replaying_transition(
    tmp_path,
) -> None:
    model = ToySpeechT5()
    controller = GradualUnfreezeController(_approach())
    controller.apply(model, global_step=50)
    callback = GradualUnfreezeCallback(
        tmp_path,
        controller,
        head_learning_rate=1e-6,
        decoder_learning_rate=5e-7,
    )

    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=50),
        SimpleNamespace(),
        model=model,
        optimizer=None,
    )

    inventory = parameter_inventory(model)
    assert callback.current_phase == "phase_2_top_decoder"
    assert inventory["trainable_parameter_count"] > 0
    assert any(
        "wrapped_decoder.layers.5" in item["name"]
        for item in inventory["trainable"]
    )
