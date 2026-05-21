"""Phase 4 acceptance tests: losses, resumable checkpoints, training smoke runs."""
import numpy as np
import torch

from fieldnet.config import DataConfig, DeepONetConfig, FNOConfig, TrainConfig
from fieldnet.data import Normalizer
from fieldnet.losses import equilibrium_residual, relative_l2
from fieldnet.models import FNO2d
from fieldnet.train import load_checkpoint, save_checkpoint, train


# --- losses ---------------------------------------------------------------
def test_relative_l2_identical_is_zero():
    x = torch.randn(4, 3, 8, 8)
    assert relative_l2(x, x).item() < 1e-6


def test_relative_l2_known_value():
    # ||0 - 1|| / ||1|| == 1
    pred = torch.zeros(2, 3, 5, 5)
    target = torch.ones(2, 3, 5, 5)
    assert abs(relative_l2(pred, target).item() - 1.0) < 1e-4


def test_relative_l2_mask_restricts_to_material():
    target = torch.ones(1, 1, 4, 4)
    pred = target.clone()
    pred[0, 0, 0, 0] = 5.0                       # error at one cell ...
    mask = torch.ones(1, 4, 4)
    mask[0, 0, 0] = 0.0                          # ... which is masked out
    assert relative_l2(pred, target, mask).item() < 1e-6


def test_equilibrium_residual_zero_for_linear_field():
    H = 16
    ax = torch.linspace(-1.0, 1.0, H)
    X, Y = torch.meshgrid(ax, ax, indexing="xy")
    # a linear displacement field -> constant stress -> zero divergence
    u = torch.stack([0.3 * X + 0.1 * Y, -0.2 * X + 0.4 * Y]).unsqueeze(0)
    res = equilibrium_residual(u, torch.ones(1, H, H), 2.0 / (H - 1), 1.0, 0.3)
    assert res.item() < 1e-8


# --- checkpointing --------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    cfg = FNOConfig(modes=4, width=8, n_layers=2)
    model = FNO2d.from_config(cfg)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    norm = Normalizer(np.zeros(3), np.ones(3), 0.0, 1.0, np.zeros(3), np.ones(3))

    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, opt, sched, epoch=7, best_metric=0.12,
                    model_kind="fno", model_cfg=cfg, normalizer=norm)
    ck = load_checkpoint(path)
    assert ck["epoch"] == 7
    assert ck["best_metric"] == 0.12
    assert ck["model_cfg"]["modes"] == 4

    restored = FNO2d.from_config(FNOConfig(**ck["model_cfg"]))
    restored.load_state_dict(ck["model_state"])
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)


# --- training smoke runs --------------------------------------------------
def test_train_fno_smoke(tiny_dataset, tmp_path):
    summary = train("fno", tiny_dataset, DataConfig(),
                    FNOConfig(modes=4, width=8, n_layers=2),
                    TrainConfig(epochs=3, batch_size=8, early_stop_patience=99),
                    tmp_path, device="cpu")
    assert summary["last_epoch"] == 2
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert np.isfinite(summary["best_sigma_vm_rel_l2"])


def test_train_deeponet_smoke(tiny_dataset, tmp_path):
    summary = train("deeponet", tiny_dataset, DataConfig(n_query_points=64),
                    DeepONetConfig(branch_layers=[3, 32, 32],
                                   trunk_layers=[2, 32, 32], n_basis=32),
                    TrainConfig(epochs=3, batch_size=8, early_stop_patience=99),
                    tmp_path, device="cpu")
    assert summary["last_epoch"] == 2
    assert (tmp_path / "last.pt").exists()
    assert np.isfinite(summary["best_sigma_vm_rel_l2"])


def test_train_resume_continues_epochs(tiny_dataset, tmp_path):
    """Acceptance: training resumes from the last checkpoint."""
    data_cfg, model_cfg = DataConfig(), FNOConfig(modes=4, width=8, n_layers=2)
    train("fno", tiny_dataset, data_cfg, model_cfg,
          TrainConfig(epochs=3, batch_size=8, early_stop_patience=99),
          tmp_path, device="cpu")
    summary = train("fno", tiny_dataset, data_cfg, model_cfg,
                    TrainConfig(epochs=6, batch_size=8, early_stop_patience=99),
                    tmp_path, resume=True, device="cpu")
    assert summary["last_epoch"] == 5            # continued 3, 4, 5
    assert load_checkpoint(tmp_path / "last.pt")["epoch"] == 5
