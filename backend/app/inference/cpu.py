"""Configure PyTorch's process-wide CPU pools once, never during inference."""

from threading import Lock
from typing import Any

_guard = Lock()
_configured_threads: int | None = None


def configure_torch_threads(torch: Any, threads: int | None) -> None:
    global _configured_threads
    with _guard:
        requested = threads or _configured_threads or 1
        if _configured_threads is not None:
            if requested != _configured_threads:
                raise RuntimeError(
                    f"PyTorch CPU threads are already {_configured_threads}; requested {requested}. "
                    "Use the same product and experiment thread setting in this process."
                )
            return
        if requested < 1:
            raise ValueError("PyTorch CPU threads must be positive")
        torch.set_num_threads(requested)
        torch.set_num_interop_threads(1)
        _configured_threads = requested
