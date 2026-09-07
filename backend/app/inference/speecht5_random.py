"""Request-local dropout for the pinned SpeechT5 decoder; no copied weights."""

from contextvars import ContextVar
from importlib.metadata import version
from inspect import signature
from types import MethodType
from typing import Any

_generator: ContextVar[Any] = ContextVar("speecht5_generator")


def install_request_dropout(model: Any, torch: Any) -> None:
    """Bind once at construction; every invocation reads its own request context."""
    if version("transformers") != "4.57.6":
        raise RuntimeError("Concurrent SpeechT5 requires the pinned transformers==4.57.6 runtime.")
    prenet = model.speecht5.decoder.prenet
    if tuple(signature(prenet._consistent_dropout).parameters) != ("inputs_embeds", "p"):
        raise RuntimeError("Unsupported SpeechT5 prenet dropout interface.")

    def dropout(_self: Any, inputs_embeds: Any, p: float) -> Any:
        # Match transformers 4.57.6, including its batch-consistent mask.
        mask = torch.bernoulli(inputs_embeds[0], p=p, generator=_generator.get())
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    prenet._consistent_dropout = MethodType(dropout, prenet)
