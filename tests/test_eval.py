"""Phase 5 acceptance tests: metrics and the end-to-end benchmark."""
import numpy as np
import torch

from fieldnet.config import DataConfig, DeepONetConfig, FNOConfig, TrainConfig
from fieldnet.eval.benchmark import run_benchmark
from fieldnet.eval.metrics import peak_stress_error, relative_l2
from fieldnet.eval.timing import time_surrogate_throughput
from fieldnet.models import FNO2d
from fieldnet.train import train


def test_relative_l2_perfect_and_known():
    mask = np.ones((4, 4))
    field = np.ones((4, 4))
    assert relative_l2(field, field, mask) < 1e-9
    # ||0 - 1|| / ||1|| == 1
    assert abs(relative_l2(np.zeros((4, 4)), field, mask) - 1.0) < 1e-6


def test_relative_l2_ignores_hole():
    target = np.ones((4, 4))
    pred = target.copy()
    pred[0, 0] = 9.0                           # error only at a masked cell
    mask = np.ones((4, 4))
    mask[0, 0] = 0.0
    assert relative_l2(pred, target, mask) < 1e-9


def test_peak_stress_error():
    true = np.zeros((4, 4)); true[1, 1] = 10.0
    pred = np.zeros((4, 4)); pred[2, 2] = 8.0
    assert abs(peak_stress_error(pred, true, np.ones((4, 4))) - 0.2) < 1e-6


def test_run_benchmark_smoke(tiny_dataset, tmp_path):
    """Acceptance: benchmark produces a results table and the three figures."""
    data_cfg = DataConfig(grid_resolution=16, mesh_n_radial=8, mesh_n_angular=16,
                          n_query_points=64)
    train("fno", tiny_dataset, data_cfg, FNOConfig(modes=4, width=8, n_layers=2),
          TrainConfig(epochs=1, batch_size=8, early_stop_patience=99),
          tmp_path / "fno", device="cpu")
    train("deeponet", tiny_dataset, data_cfg,
          DeepONetConfig(branch_layers=[3, 16, 16], trunk_layers=[2, 16, 16],
                         n_basis=16),
          TrainConfig(epochs=1, batch_size=8, early_stop_patience=99),
          tmp_path / "deeponet", device="cpu")

    out = tmp_path / "bench"
    results = run_benchmark(
        {"fno": tmp_path / "fno" / "best.pt",
         "deeponet": tmp_path / "deeponet" / "best.pt"},
        tiny_dataset, data_cfg, out, n_timing_runs=2, device="cpu")

    for name in ("results.json", "results.md", "accuracy.png",
                 "speedup.png", "triptych.png"):
        assert (out / name).exists(), name
    assert set(results["accuracy"]) == {"fno", "deeponet"}
    assert results["timing"]["fem_solve"]["mean_ms"] > 0.0
    for kind in ("fno", "deeponet"):
        assert np.isfinite(results["accuracy"][kind]["ood"]["sigma_vm"])
        assert results["speedup"][f"{kind}_latency_cpu"] > 0.0


def test_throughput_timing_amortizes_per_field():
    """Batched throughput reports a positive per-field wall-clock time."""
    model = FNO2d(modes=4, width=8, n_layers=2, in_channels=4, out_channels=3)
    args = (torch.randn(1, 4, 16, 16),)
    stats = time_surrogate_throughput(model, "fno", args, batch_size=4,
                                      n_runs=2, device=torch.device("cpu"))
    assert stats["mean_ms"] > 0.0
