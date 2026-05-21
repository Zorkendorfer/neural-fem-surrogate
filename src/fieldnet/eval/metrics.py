"""Accuracy metrics for the surrogate-vs-FEM benchmark."""
import numpy as np
from tqdm import tqdm

from fieldnet.eval.inference import predict_grid

CHANNELS = ["u_x", "u_y", "sigma_vm"]


def relative_l2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Relative L2 error over material pixels: ``||pred-target|| / ||target||``."""
    m = mask.astype(bool)
    diff = np.sqrt(np.sum((pred[m] - target[m]) ** 2))
    denom = np.sqrt(np.sum(target[m] ** 2))
    return float(diff / (denom + 1e-12))


def peak_stress_error(pred_vm: np.ndarray, true_vm: np.ndarray,
                      mask: np.ndarray) -> float:
    """Relative error of the peak von Mises stress (stress-concentration proxy)."""
    m = mask.astype(bool)
    peak_true = float(true_vm[m].max())
    return abs(float(pred_vm[m].max()) - peak_true) / (peak_true + 1e-12)


def evaluate_split(model, kind, raw, indices, norm, device="cpu",
                   desc=None) -> dict:
    """Mean per-channel relative L2, displacement rel L2, and peak-stress error
    over the samples in ``indices`` (all in physical units)."""
    per = {c: [] for c in CHANNELS}
    disp, peak = [], []
    for idx in tqdm(indices, desc=desc or f"eval {kind}"):
        pred = predict_grid(model, kind, raw, idx, norm, device)   # (3, H, W)
        true = raw["fields"][idx]
        mask = raw["mask"][idx]
        for c, name in enumerate(CHANNELS):
            per[name].append(relative_l2(pred[c], true[c], mask))
        mask2 = np.broadcast_to(mask, (2,) + mask.shape)
        disp.append(relative_l2(pred[:2], true[:2], mask2))
        peak.append(peak_stress_error(pred[2], true[2], mask))
    return {
        "sigma_vm": float(np.mean(per["sigma_vm"])),
        "u_x": float(np.mean(per["u_x"])),
        "u_y": float(np.mean(per["u_y"])),
        "displacement": float(np.mean(disp)),
        "peak_stress_error": float(np.mean(peak)),
        "n_samples": int(len(indices)),
    }
