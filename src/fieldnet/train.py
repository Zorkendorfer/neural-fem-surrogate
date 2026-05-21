"""Training loop for the FNO and DeepONet operators.

Trains in normalized space with a relative-L2 field loss; validation metrics
are reported in physical units (per channel). Checkpoints are resumable and an
optional physics-consistency term is gated behind a config weight.
"""
import csv
import warnings
from pathlib import Path

import torch

from fieldnet.config import DataConfig, DeepONetConfig, FNOConfig, TrainConfig
from tqdm import tqdm
from fieldnet.data import build_dataloaders
from fieldnet.losses import equilibrium_residual, relative_l2
from fieldnet.models import DeepONet, FNO2d
from fieldnet.utils.device import get_device
from fieldnet.utils.logging import get_logger
from fieldnet.utils.seed import set_seed

logger = get_logger("fieldnet.train")
FIELD_NAMES = ["u_x", "u_y", "sigma_vm"]


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric,
                    model_kind, model_cfg, normalizer) -> None:
    """Persist everything needed to resume training or to serve the model."""
    torch.save({
        "epoch": epoch,
        "best_metric": best_metric,
        "model_kind": model_kind,
        "model_cfg": model_cfg.model_dump(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "norm": {
            "field_mean": normalizer.field_mean,
            "field_std": normalizer.field_std,
            "sdf_mean": normalizer.sdf_mean,
            "sdf_std": normalizer.sdf_std,
            "theta_lo": normalizer.theta_lo,
            "theta_hi": normalizer.theta_hi,
        },
    }, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint dict (trusted file -> weights_only=False)."""
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------
# metric logging (CSV always; W&B optional)
# --------------------------------------------------------------------------
class MetricLogger:
    """Append per-epoch metrics to a CSV; mirror to W&B when enabled."""

    def __init__(self, csv_path, use_wandb: bool, run_name: str, config: dict):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_done = False
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project="fieldnet", name=run_name, config=config)
                self.wandb = wandb
            except Exception as exc:               # missing pkg / not logged in
                warnings.warn(f"W&B logging disabled: {exc}")

    def log(self, row: dict) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not self._header_done:
                writer.writeheader()
                self._header_done = True
            writer.writerow(row)
        if self.wandb is not None:
            self.wandb.log(row)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


# --------------------------------------------------------------------------
# model / batch helpers
# --------------------------------------------------------------------------
def _build_model(model_kind: str, model_cfg):
    return FNO2d.from_config(model_cfg) if model_kind == "fno" \
        else DeepONet.from_config(model_cfg)


def _select_device(model_kind: str, requested):
    device = get_device(requested or "auto")
    if model_kind == "fno" and device.type == "mps":
        warnings.warn("FNO relies on torch.fft; MPS support is limited -> CPU")
        return torch.device("cpu")
    return device


def _forward(model_kind, model, batch, device):
    """Return (pred, target, mask) in normalized space; mask is None for DeepONet."""
    if model_kind == "fno":
        pred = model(batch["input"].to(device))
        return pred, batch["target"].to(device), batch["mask"].to(device)
    pred = model(batch["theta"].to(device), batch["coords"].to(device))
    return pred, batch["target"].to(device), None


def _train_epoch(model_kind, model, loader, optimizer, device, train_cfg, phys,
                 epoch: int):
    model.train()
    total, count = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} (Train)")
    for batch in pbar:
        pred, target, mask = _forward(model_kind, model, batch, device)
        loss = relative_l2(pred, target, mask)
        if phys is not None:
            loss = loss + train_cfg.physics_loss_weight * phys(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        bs = pred.shape[0]
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def _evaluate(model_kind, model, loader, device, fmean, fstd):
    """Per-channel relative L2 in physical units (denormalized)."""
    model.eval()
    if model_kind == "fno":
        mb, sb = fmean.view(1, 3, 1, 1), fstd.view(1, 3, 1, 1)
    else:
        mb, sb = fmean.view(1, 1, 3), fstd.view(1, 1, 3)
    per = {name: [] for name in FIELD_NAMES}
    for batch in loader:
        pred, target, mask = _forward(model_kind, model, batch, device)
        pred_p, target_p = pred * sb + mb, target * sb + mb
        for c, name in enumerate(FIELD_NAMES):
            if model_kind == "fno":
                rel = relative_l2(pred_p[:, c], target_p[:, c], mask, reduce=False)
            else:
                rel = relative_l2(pred_p[..., c], target_p[..., c], reduce=False)
            per[name].append(rel)
    return {name: torch.cat(vals).mean().item() for name, vals in per.items()}


# --------------------------------------------------------------------------
# training entrypoint
# --------------------------------------------------------------------------
def train(model_kind: str, dataset_path, data_cfg: DataConfig,
          model_cfg, train_cfg: TrainConfig, out_dir,
          resume: bool = False, use_wandb: bool = False, device=None) -> dict:
    """Train one operator; return a summary dict. Writes last/best checkpoints."""
    if model_kind not in ("fno", "deeponet"):
        raise ValueError(f"unknown model_kind {model_kind!r}")
    set_seed(data_cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(model_kind, device)
    logger.info("training %s on %s", model_kind, device)

    view = "fno" if model_kind == "fno" else "deeponet"
    loaders, normalizer, _ = build_dataloaders(
        dataset_path, data_cfg, view=view, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers)

    model = _build_model(model_kind, model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                                 weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.epochs, 1), eta_min=train_cfg.min_lr)

    fmean = torch.tensor(normalizer.field_mean, dtype=torch.float32, device=device)
    fstd = torch.tensor(normalizer.field_std, dtype=torch.float32, device=device)

    phys = None
    if model_kind == "fno" and train_cfg.physics_loss_weight > 0:
        dx = 2.0 * data_cfg.half_width / (data_cfg.grid_resolution - 1)

        def phys(pred, mask, _dx=dx):
            u = pred[:, :2] * fstd[:2].view(1, 2, 1, 1) + fmean[:2].view(1, 2, 1, 1)
            return equilibrium_residual(u, mask, _dx, data_cfg.youngs_modulus,
                                        data_cfg.poisson_ratio)

    start_epoch, best = 0, float("inf")
    no_improve = 0
    last_ckpt, best_ckpt = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_ckpt.exists():
        ck = load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck["scheduler_state"]:
            scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best = ck["best_metric"]
        logger.info("resumed from checkpoint at epoch %d", start_epoch)

    mlogger = MetricLogger(out_dir / "metrics.csv", use_wandb,
                           run_name=model_kind,
                           config={"model": model_kind, **train_cfg.model_dump()})

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, train_cfg.epochs):
        train_loss = _train_epoch(model_kind, model, loaders["train"],
                                  optimizer, device, train_cfg, phys, epoch)
        val = _evaluate(model_kind, model, loaders["val"], device, fmean, fstd)
        scheduler.step()
        last_epoch = epoch

        metric = val["sigma_vm"]
        mlogger.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_rel_l2_{n}": round(val[n], 6) for n in FIELD_NAMES},
        })
        logger.info("epoch %d  train_loss=%.4f  val sigma_vm rel L2=%.4f",
                    epoch, train_loss, metric)

        if metric < best:
            best, no_improve = metric, 0
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best,
                            model_kind, model_cfg, normalizer)
        else:
            no_improve += 1
        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best,
                        model_kind, model_cfg, normalizer)

        if no_improve >= train_cfg.early_stop_patience:
            logger.info("early stopping at epoch %d (no improvement)", epoch)
            break

    mlogger.finish()
    logger.info("done. best val sigma_vm rel L2 = %.4f", best)
    return {"model_kind": model_kind, "last_epoch": last_epoch,
            "best_sigma_vm_rel_l2": best}
