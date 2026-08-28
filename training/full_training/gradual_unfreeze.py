from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from transformers import Seq2SeqTrainer, TrainerCallback


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    trainable = []
    frozen = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        record = {"name": name, "parameters": parameter.numel()}
        if parameter.requires_grad:
            trainable.append(record)
            trainable_parameters += parameter.numel()
        else:
            frozen.append(record)
            frozen_parameters += parameter.numel()
    return {
        "trainable_parameter_count": trainable_parameters,
        "frozen_parameter_count": frozen_parameters,
        "total_parameter_count": trainable_parameters + frozen_parameters,
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable": trainable,
        "frozen": frozen,
    }


class GradualUnfreezeController:
    """Own the deterministic two-phase SpeechT5 trainability contract."""

    def __init__(self, approach: dict[str, Any]):
        self.transition_after_step = int(approach["transition_after_step"])
        self.head_prefixes = tuple(approach["phase_one_trainable_prefixes"])
        self.decoder_prefixes = tuple(approach["phase_two_decoder_prefixes"])
        if self.transition_after_step <= 0:
            raise ValueError("transition_after_step must be positive")
        if not self.head_prefixes or not self.decoder_prefixes:
            raise ValueError("gradual unfreezing requires head and decoder prefixes")

    def role_for_parameter(self, name: str) -> str | None:
        if _matches(name, self.head_prefixes):
            return "heads"
        if _matches(name, self.decoder_prefixes):
            return "top_decoder"
        return None

    def apply(self, model: torch.nn.Module, *, global_step: int) -> str:
        phase = "phase_2_top_decoder" if global_step >= self.transition_after_step else "phase_1_heads"
        for name, parameter in model.named_parameters():
            role = self.role_for_parameter(name)
            parameter.requires_grad = role == "heads" or (
                role == "top_decoder" and phase == "phase_2_top_decoder"
            )
        inventory = parameter_inventory(model)
        if inventory["trainable_parameter_count"] <= 0:
            raise RuntimeError("gradual-unfreeze policy left no trainable parameters")
        return phase

    def optimizer_parameters(
        self, model: torch.nn.Module
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        heads: list[torch.nn.Parameter] = []
        decoder: list[torch.nn.Parameter] = []
        names_seen: set[str] = set()
        for name, parameter in model.named_parameters():
            role = self.role_for_parameter(name)
            if role == "heads":
                heads.append(parameter)
                names_seen.add(name)
            elif role == "top_decoder":
                decoder.append(parameter)
                names_seen.add(name)
        if not heads or not decoder:
            raise RuntimeError(
                "configured gradual-unfreeze prefixes did not resolve both optimizer roles"
            )
        expected = {
            name
            for name, _ in model.named_parameters()
            if self.role_for_parameter(name) is not None
        }
        if names_seen != expected:
            raise RuntimeError("optimizer parameter inventory is incomplete")
        return heads, decoder


class GradualUnfreezeTrainer(Seq2SeqTrainer):
    """Trainer with stable, explicit optimizer groups across the phase transition."""

    def __init__(
        self,
        *args,
        unfreeze_controller: GradualUnfreezeController,
        head_learning_rate: float,
        decoder_learning_rate: float,
        train_sampler_factory=None,
        **kwargs,
    ):
        self.unfreeze_controller = unfreeze_controller
        self.head_learning_rate = float(head_learning_rate)
        self.decoder_learning_rate = float(decoder_learning_rate)
        self.train_sampler_factory = train_sampler_factory
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        if self.train_dataset is None or self.train_sampler_factory is None:
            return super()._get_train_sampler()
        return self.train_sampler_factory(len(self.train_dataset))

    def create_optimizer(self) -> Optimizer:
        if self.optimizer is None:
            heads, decoder = self.unfreeze_controller.optimizer_parameters(self.model)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
            self.optimizer = optimizer_cls(
                [
                    {
                        "params": heads,
                        "lr": self.head_learning_rate,
                        "initial_lr": self.head_learning_rate,
                        "group_name": "modal_tts_heads",
                    },
                    {
                        "params": decoder,
                        "lr": self.decoder_learning_rate,
                        "initial_lr": self.decoder_learning_rate,
                        "group_name": "top_two_decoder_blocks",
                    },
                ],
                **optimizer_kwargs,
            )
        return self.optimizer


class GradualUnfreezeCallback(TrainerCallback):
    def __init__(
        self,
        output: Path,
        controller: GradualUnfreezeController,
        *,
        head_learning_rate: float,
        decoder_learning_rate: float,
    ):
        self.output = output
        self.controller = controller
        self.head_learning_rate = float(head_learning_rate)
        self.decoder_learning_rate = float(decoder_learning_rate)
        self.current_phase: str | None = None
        self.events: list[dict[str, Any]] = []

    def _record(self, model, optimizer, global_step: int, reason: str) -> None:
        phase = self.controller.apply(model, global_step=global_step)
        groups = []
        if optimizer is not None:
            groups = [
                {
                    "group_name": group.get("group_name"),
                    "configured_initial_lr": group.get("initial_lr"),
                    "current_lr": group.get("lr"),
                    "parameter_tensors": len(group["params"]),
                    "parameters": sum(value.numel() for value in group["params"]),
                }
                for group in optimizer.param_groups
            ]
        event = {
            "recorded_utc": _utc(),
            "global_step": int(global_step),
            "phase": phase,
            "reason": reason,
            "optimizer_groups": groups,
            "parameter_inventory": parameter_inventory(model),
        }
        if not self.events or (
            self.events[-1]["global_step"], self.events[-1]["phase"], self.events[-1]["reason"]
        ) != (event["global_step"], event["phase"], event["reason"]):
            self.events.append(event)
        self.current_phase = phase
        _atomic_json(
            self.output / "parameter_inventory.json",
            {
                "schema_version": 1,
                "transition_after_step": self.controller.transition_after_step,
                "head_learning_rate": self.head_learning_rate,
                "decoder_learning_rate": self.decoder_learning_rate,
                "current_phase": phase,
                "events": self.events,
            },
        )

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self._record(model, optimizer, state.global_step, "train_begin_or_resume")
        return control

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if (
            state.global_step >= self.controller.transition_after_step
            and self.current_phase != "phase_2_top_decoder"
        ):
            self._record(model, optimizer, state.global_step, "scheduled_transition")
        return control

    def on_save(self, args, state, control, model=None, optimizer=None, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        phase = self.controller.apply(model, global_step=state.global_step)
        _atomic_json(
            checkpoint / "gradual_unfreeze_state.json",
            {
                "schema_version": 1,
                "checkpoint_step": state.global_step,
                "transition_after_step": self.controller.transition_after_step,
                "phase": phase,
                "head_learning_rate": self.head_learning_rate,
                "decoder_learning_rate": self.decoder_learning_rate,
                "parameter_inventory": parameter_inventory(model),
                "optimizer_groups": [
                    {
                        "group_name": group.get("group_name"),
                        "configured_initial_lr": group.get("initial_lr"),
                        "current_lr": group.get("lr"),
                        "parameter_tensors": len(group["params"]),
                        "parameters": sum(value.numel() for value in group["params"]),
                    }
                    for group in (optimizer.param_groups if optimizer is not None else [])
                ],
                "written_utc": _utc(),
            },
        )
        return control
