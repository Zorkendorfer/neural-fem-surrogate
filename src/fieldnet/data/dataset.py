"""PyTorch Dataset/DataLoader views for the FNO (grid) and DeepONet (point) models."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fieldnet.config import DataConfig
from fieldnet.data.normalization import Normalizer
from fieldnet.data.splits import make_splits


def load_raw(npz_path) -> dict:
    """Load the full dataset into memory as plain numpy arrays."""
    with np.load(npz_path) as d:
        return {k: d[k] for k in d.files}


def _grid_coords(axis: np.ndarray) -> np.ndarray:
    """Flat (H*W, 2) physical coordinates, row-major to match field raveling."""
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)


class FNODataset(Dataset):
    """Grid view: input channels ``[SDF, mask, sigma_inf, alpha]`` -> field grid.

    The hole radius ``r`` is encoded implicitly by the SDF and mask; the scalar
    loads are broadcast to constant channels. The target field is zeroed inside
    the hole so a masked loss is straightforward.
    """

    def __init__(self, raw: dict, indices: np.ndarray, normalizer: Normalizer):
        self.fields = raw["fields"]
        self.sdf = raw["sdf"]
        self.mask = raw["mask"]
        self.theta = raw["theta"]
        self.indices = np.asarray(indices)
        self.norm = normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        mask = self.mask[j].astype(np.float32)                       # (H, W)
        sdf = self.norm.norm_sdf(self.sdf[j]).astype(np.float32)
        theta_n = self.norm.norm_theta(self.theta[j]).astype(np.float32)
        H, W = mask.shape
        sig = np.full((H, W), theta_n[1], np.float32)
        alpha = np.full((H, W), theta_n[2], np.float32)
        inp = np.stack([sdf, mask, sig, alpha], axis=0)              # (4, H, W)
        target = self.norm.norm_fields(self.fields[j]).astype(np.float32) * mask
        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),                      # (3, H, W)
            "mask": torch.from_numpy(mask),
            "theta": torch.from_numpy(theta_n),
        }


class DeepONetDataset(Dataset):
    """Point view: parameter vector (branch) + sampled query coords (trunk).

    Query points are drawn from the material domain only; sampling is
    deterministic per sample (seed + index) for reproducible batches.
    """

    def __init__(self, raw: dict, indices: np.ndarray, normalizer: Normalizer,
                 coords_flat: np.ndarray, n_query: int, seed: int):
        self.fields = raw["fields"]
        self.mask = raw["mask"]
        self.theta = raw["theta"]
        self.indices = np.asarray(indices)
        self.norm = normalizer
        self.coords_flat = coords_flat                               # (H*W, 2)
        self.n_query = n_query
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        material = np.where(self.mask[j].astype(bool).ravel())[0]
        rng = np.random.default_rng(self.seed + j)
        sel = rng.choice(material, size=self.n_query,
                         replace=len(material) < self.n_query)
        coords = self.coords_flat[sel]                               # (n_query, 2)
        fields_flat = self.norm.norm_fields(self.fields[j]).reshape(3, -1).T
        target = fields_flat[sel].astype(np.float32)                 # (n_query, 3)
        theta_n = self.norm.norm_theta(self.theta[j]).astype(np.float32)
        return {
            "theta": torch.from_numpy(theta_n),
            "coords": torch.from_numpy(coords),
            "target": torch.from_numpy(target),
        }


def build_dataloaders(npz_path, cfg: DataConfig, view: str = "fno",
                      batch_size: int = 16, n_query: int | None = None,
                      num_workers: int = 0, normalizer: Normalizer | None = None):
    """Build ``{train, val, test, ood}`` DataLoaders for one model view.

    Returns ``(loaders, normalizer, splits)``. The normalizer is fit on the
    train split unless one is supplied (e.g. loaded for inference).
    """
    if view not in ("fno", "deeponet"):
        raise ValueError(f"unknown view {view!r} (expected 'fno' or 'deeponet')")

    raw = load_raw(npz_path)
    splits = make_splits(raw["theta"], cfg)
    if normalizer is None:
        normalizer = Normalizer.fit(raw["fields"], raw["sdf"], raw["mask"],
                                    splits.train, cfg)
    n_query = n_query if n_query is not None else cfg.n_query_points
    coords_flat = _grid_coords(raw["coords"])

    loaders = {}
    for name, idx in splits.as_dict().items():
        if view == "fno":
            ds: Dataset = FNODataset(raw, idx, normalizer)
        else:
            ds = DeepONetDataset(raw, idx, normalizer, coords_flat, n_query, cfg.seed)
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=num_workers, drop_last=False)
    return loaders, normalizer, splits
