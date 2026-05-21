"""Generate the FEM ground-truth dataset.

Examples:
    python scripts/generate.py                          # n_samples from config
    python scripts/generate.py --n-samples 200 --workers 8
"""
import argparse
from pathlib import Path

from fieldnet.config import DataConfig, load_config
from fieldnet.fem.generate import generate_dataset
from fieldnet.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate FEM dataset")
    parser.add_argument("--config", type=Path, default="configs/data.yaml")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="override config n_samples")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel FEM solve processes")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .npz path (default: <data_dir>/dataset.npz)")
    args = parser.parse_args()

    cfg = load_config(args.config, DataConfig)
    set_seed(cfg.seed)
    generate_dataset(cfg, n_samples=args.n_samples, workers=args.workers,
                     out_path=args.out)


if __name__ == "__main__":
    main()
