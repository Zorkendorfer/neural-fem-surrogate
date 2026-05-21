"""Inspect the Phase 2 data pipeline: split sizes, batch shapes, normalization.

Example:
    python scripts/inspect_data.py --dataset data/dataset.npz
"""
import argparse
from pathlib import Path

import numpy as np

from fieldnet.config import DataConfig, load_config
from fieldnet.data import build_dataloaders
from fieldnet.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Inspect the data pipeline")
    parser.add_argument("--config", type=Path, default="configs/data.yaml")
    parser.add_argument("--dataset", type=Path, default="data/dataset.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-stats", type=Path, default="data/norm_stats.npz")
    args = parser.parse_args()

    cfg = load_config(args.config, DataConfig)
    set_seed(cfg.seed)

    splits = None
    for view in ("fno", "deeponet"):
        loaders, norm, splits = build_dataloaders(
            args.dataset, cfg, view=view, batch_size=args.batch_size)
        sizes = {k: len(v.dataset) for k, v in loaders.items()}
        print(f"\n[{view}] split sizes: {sizes}")
        batch = next(iter(loaders["train"]))
        for key, val in batch.items():
            print(f"  {key:8s} {tuple(val.shape)}  {val.dtype}")
        if view == "fno":
            norm.save(args.save_stats)
            with np.printoptions(precision=3):
                print(f"  field_mean={norm.field_mean}  field_std={norm.field_std}")
            print(f"  saved normalization stats -> {args.save_stats}")

    with np.load(args.dataset) as d:
        r = d["theta"][:, 0]
    print(f"\ngeometry split: train r-max={r[splits.train].max():.4f}  "
          f"ood r-min={r[splits.ood].min():.4f}  -> zero overlap")


if __name__ == "__main__":
    main()
