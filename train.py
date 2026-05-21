"""Main training entry point."""
import argparse
import yaml
from pathlib import Path

import torch
from fieldnet.config import DataConfig, FNOConfig, DeepONetConfig
from fieldnet.data.dataset import build_dataloaders
from fieldnet.models.fno import FNO2d
from fieldnet.models.deeponet import DeepONet
from fieldnet.train import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train FieldNet surrogate")
    parser.add_argument("--model", type=str, choices=["fno", "deeponet"], required=True)
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Load configs
    with open(args.data_config, "r") as f:
        data_cfg = DataConfig(**yaml.safe_load(f))
    
    with open(args.model_config, "r") as f:
        m_cfg_dict = yaml.safe_load(f)

    # Setup Data
    view = "fno" if args.model == "fno" else "deeponet"
    loaders, normalizer, _ = build_dataloaders(
        data_cfg.data_dir / "dataset.npz", 
        data_cfg, 
        view=view, 
        batch_size=args.batch_size
    )

    # Setup Model
    if args.model == "fno":
        m_cfg = FNOConfig(**m_cfg_dict)
        model = FNO2d.from_config(m_cfg)
    else:
        m_cfg = DeepONetConfig(**m_cfg_dict)
        model = DeepONet.from_config(m_cfg)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=f"checkpoints/{args.model}",
        use_wandb=not args.no_wandb,
        config={
            "model_type": args.model,
            "lr": args.lr,
            "batch_size": args.batch_size,
            **m_cfg_dict,
            **data_cfg.dict()
        }
    )

    # Train
    try:
        trainer.fit(epochs=args.epochs)
    except KeyboardInterrupt:
        print("Training interrupted. Best model saved.")

if __name__ == "__main__":
    main()