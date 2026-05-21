"""Deterministic train/val/test splits with an out-of-distribution geometry holdout."""
from dataclasses import dataclass

import numpy as np

from fieldnet.config import DataConfig


@dataclass
class Splits:
    """Index arrays into the dataset, one per partition."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    ood: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val,
                "test": self.test, "ood": self.ood}


def make_splits(theta: np.ndarray, cfg: DataConfig) -> Splits:
    """Partition samples by hole radius.

    Samples with ``r > ood_r_threshold`` form the out-of-distribution geometry
    holdout; the rest are shuffled with a fixed seed into train/val/test by the
    config fractions. The OOD set therefore shares no geometry with train *by
    construction* -- the headline "generalizes to unseen geometries" claim.
    """
    r = theta[:, 0]
    ood = np.where(r > cfg.ood_r_threshold)[0]
    indist = np.where(r <= cfg.ood_r_threshold)[0]

    perm = np.random.default_rng(cfg.seed).permutation(indist)
    n = len(perm)
    n_train = int(round(n * cfg.train_fraction))
    n_val = int(round(n * cfg.val_fraction))

    return Splits(
        train=np.sort(perm[:n_train]),
        val=np.sort(perm[n_train:n_train + n_val]),
        test=np.sort(perm[n_train + n_val:]),
        ood=np.sort(ood),
    )
