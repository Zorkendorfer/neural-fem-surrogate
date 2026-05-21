"""Per-channel normalization statistics, fit on the train split and persisted."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fieldnet.config import DataConfig


@dataclass
class Normalizer:
    """Standardizes fields/SDF and min-max scales the parameter vector.

    Field and SDF stats are fit on the train split only (no val/test/OOD
    leakage). Parameter scaling uses the known config ranges, so it needs no
    data and stays valid at inference time.
    """
    field_mean: np.ndarray   # (3,)
    field_std: np.ndarray    # (3,)
    sdf_mean: float
    sdf_std: float
    theta_lo: np.ndarray     # (3,)
    theta_hi: np.ndarray     # (3,)

    @classmethod
    def fit(cls, fields: np.ndarray, sdf: np.ndarray, mask: np.ndarray,
            train_idx: np.ndarray, cfg: DataConfig) -> "Normalizer":
        """Fit on the train split; field stats use material pixels only."""
        m = mask[train_idx].astype(bool)                  # (n, H, W)
        fmean = np.empty(3, np.float64)
        fstd = np.empty(3, np.float64)
        for c in range(3):
            vals = fields[train_idx, c][m]
            fmean[c] = vals.mean()
            fstd[c] = vals.std() + 1e-8
        s = sdf[train_idx]
        theta_lo = np.array([cfg.r_range[0], cfg.sigma_inf_range[0],
                             cfg.alpha_range[0]], np.float64)
        theta_hi = np.array([cfg.r_range[1], cfg.sigma_inf_range[1],
                             cfg.alpha_range[1]], np.float64)
        return cls(fmean, fstd, float(s.mean()), float(s.std() + 1e-8),
                   theta_lo, theta_hi)

    # --- fields (channel axis is -3, broadcasts over single or batched) ----
    def norm_fields(self, x):
        return (x - self.field_mean.reshape(3, 1, 1)) / self.field_std.reshape(3, 1, 1)

    def denorm_fields(self, x):
        return x * self.field_std.reshape(3, 1, 1) + self.field_mean.reshape(3, 1, 1)

    # --- sdf ---------------------------------------------------------------
    def norm_sdf(self, x):
        return (x - self.sdf_mean) / self.sdf_std

    # --- theta -> [0, 1] ---------------------------------------------------
    def norm_theta(self, theta):
        return (theta - self.theta_lo) / (self.theta_hi - self.theta_lo)

    def denorm_theta(self, theta_n):
        return theta_n * (self.theta_hi - self.theta_lo) + self.theta_lo

    # --- io ----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, field_mean=self.field_mean, field_std=self.field_std,
                 sdf_mean=self.sdf_mean, sdf_std=self.sdf_std,
                 theta_lo=self.theta_lo, theta_hi=self.theta_hi)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with np.load(path) as d:
            return cls(d["field_mean"], d["field_std"], float(d["sdf_mean"]),
                       float(d["sdf_std"]), d["theta_lo"], d["theta_hi"])
