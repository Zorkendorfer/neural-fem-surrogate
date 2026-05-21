"""Evaluation metrics for benchmarking."""
import torch
from fieldnet.losses import relative_l2

def compute_metrics(pred, target, mask=None):
    """Standard evaluation suite for the surrogate."""
    return {
        "rel_l2": relative_l2(pred, target, mask).item(),
        # Peak stress error is a key 'engineering money metric' for Phase 5
        "peak_error": (pred.max() - target.max()).abs().item() / (target.max() + 1e-8)
    }