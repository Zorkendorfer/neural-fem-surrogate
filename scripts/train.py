"""Train a neural operator (FNO or DeepONet).

Examples:
    python scripts/train.py --model fno
    python scripts/train.py --model deeponet --wandb
    python scripts/train.py --model fno --resume
"""
import argparse
from pathlib import Path

from fieldnet.config import (DataConfig, DeepONetConfig, FNOConfig,
                             TrainConfig, load_config)
from fieldnet.train import train

_MODEL_CONFIG = {"fno": ("configs/fno.yaml", FNOConfig),
                 "deeponet": ("configs/deeponet.yaml", DeepONetConfig)}


def main():
    parser = argparse.ArgumentParser(description="Train FNO or DeepONet")
    parser.add_argument("--model", choices=["fno", "deeponet"], required=True)
    parser.add_argument("--dataset", type=Path, default="data/dataset.npz")
    parser.add_argument("--data-config", type=Path, default="configs/data.yaml")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default="configs/train.yaml")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu / auto")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config, DataConfig)
    train_cfg = load_config(args.train_config, TrainConfig)
    default_path, model_class = _MODEL_CONFIG[args.model]
    model_cfg = load_config(args.model_config or default_path, model_class)
    out_dir = args.out_dir or Path("checkpoints") / args.model

    summary = train(args.model, args.dataset, data_cfg, model_cfg, train_cfg,
                    out_dir, resume=args.resume, use_wandb=args.wandb,
                    device=args.device)
    print(f"best val sigma_vm rel L2: {summary['best_sigma_vm_rel_l2']:.4f}  "
          f"-> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
