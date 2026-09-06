"""One-time process CPU runtime configuration."""

import os


def configure_cpu_runtime(cpu_threads: int | None) -> None:
    """Set native thread budgets before ONNX or PyTorch sessions are created."""
    if cpu_threads is None:
        return
    value = str(max(1, cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", value)
    os.environ.setdefault("MKL_NUM_THREADS", value)
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_initialized():
        torch.set_num_threads(int(value))
        torch.set_num_interop_threads(1)
