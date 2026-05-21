"""Load trained operators and run full-grid inference."""
import numpy as np
import torch

from fieldnet.config import DeepONetConfig, FNOConfig
from fieldnet.data.dataset import _grid_coords
from fieldnet.data.normalization import Normalizer
from fieldnet.models import DeepONet, FNO2d
from fieldnet.train import load_checkpoint


def load_trained(checkpoint_path, device="cpu"):
    """Rebuild a trained model and its normalizer from a checkpoint.

    Returns ``(model, normalizer, model_kind)``.
    """
    ck = load_checkpoint(checkpoint_path, map_location=device)
    kind = ck["model_kind"]
    if kind == "fno":
        model = FNO2d.from_config(FNOConfig(**ck["model_cfg"]))
    elif kind == "deeponet":
        model = DeepONet.from_config(DeepONetConfig(**ck["model_cfg"]))
    else:
        raise ValueError(f"unknown model_kind {kind!r}")
    model.load_state_dict(ck["model_state"])
    model.eval().to(device)
    return model, Normalizer(**ck["norm"]), kind


def build_fno_input(raw: dict, idx: int, norm: Normalizer) -> np.ndarray:
    """(4, H, W) FNO input for sample ``idx``: [SDF, mask, sigma_inf, alpha]."""
    H = raw["mask"].shape[-1]
    theta_n = norm.norm_theta(raw["theta"][idx]).astype(np.float32)
    sdf = norm.norm_sdf(raw["sdf"][idx]).astype(np.float32)
    mask = raw["mask"][idx].astype(np.float32)
    sig = np.full((H, H), theta_n[1], np.float32)
    alpha = np.full((H, H), theta_n[2], np.float32)
    return np.stack([sdf, mask, sig, alpha], axis=0)


@torch.no_grad()
def predict_grid(model, kind, raw, idx, norm, device="cpu") -> np.ndarray:
    """Physical-unit field prediction ``(3, H, W)`` for sample ``idx``.

    Evaluates both model families on the full reference grid (the DeepONet is
    queried at every grid point) so the metric is identical for both.
    """
    H = raw["mask"].shape[-1]
    mask = raw["mask"][idx].astype(np.float32)
    if kind == "fno":
        inp = torch.from_numpy(build_fno_input(raw, idx, norm)).unsqueeze(0)
        out = model(inp.to(device))[0]                           # (3, H, W)
    else:
        theta_n = norm.norm_theta(raw["theta"][idx]).astype(np.float32)
        coords = _grid_coords(raw["coords"])                     # (H*W, 2)
        th = torch.from_numpy(theta_n).unsqueeze(0).to(device)
        co = torch.from_numpy(coords).unsqueeze(0).to(device)
        out = model(th, co)[0].T.reshape(3, H, H)                # (3, H, W)

    fmean = torch.tensor(norm.field_mean, dtype=torch.float32, device=device)
    fstd = torch.tensor(norm.field_std, dtype=torch.float32, device=device)
    out = out * fstd.view(3, 1, 1) + fmean.view(3, 1, 1)
    return (out.cpu().numpy() * mask).astype(np.float32)         # hole zeroed
