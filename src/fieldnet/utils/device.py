"""Compute-device selection that works on Linux/Windows GPUs, Apple Silicon, CPU."""
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return a ``torch.device``.

    ``prefer`` is ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``. ``"auto"``
    picks CUDA first (Linux/Windows NVIDIA), then Apple-Silicon MPS (M-series
    Macs), then CPU -- so the same code runs unchanged on all three platforms.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
