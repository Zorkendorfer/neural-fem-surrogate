"""The benchmark: surrogate accuracy + speedup vs the FEM ground truth."""
import json
from pathlib import Path

import numpy as np
import torch

from fieldnet.data.dataset import _grid_coords, load_raw
from fieldnet.data.splits import make_splits
from fieldnet.eval.inference import build_fno_input, load_trained
from fieldnet.eval.metrics import evaluate_split
from fieldnet.eval.plots import plot_accuracy, plot_speedup, plot_triptych
from fieldnet.eval.timing import (time_fem_solve, time_surrogate,
                                  time_surrogate_throughput)
from fieldnet.utils.device import get_device
from fieldnet.utils.logging import get_logger

logger = get_logger("fieldnet.benchmark")


def _surrogate_args(kind, raw, idx, norm):
    """Model input tuple for a single sample (batch size 1)."""
    if kind == "fno":
        return (torch.from_numpy(build_fno_input(raw, idx, norm)).unsqueeze(0),)
    theta_n = norm.norm_theta(raw["theta"][idx]).astype(np.float32)
    coords = _grid_coords(raw["coords"])
    return (torch.from_numpy(theta_n).unsqueeze(0),
            torch.from_numpy(coords).unsqueeze(0))


def _accel_for(kind, accel):
    """Accelerator usable by this model, or None (FNO needs FFT -> not MPS)."""
    if accel.type == "cpu":
        return None
    if kind == "fno" and accel.type == "mps":
        return None
    return accel


def _write_markdown(results: dict, path) -> None:
    acc = results["accuracy"]
    lines = ["# FieldNet benchmark", "",
             "## Accuracy — relative L2 vs FEM ground truth (%)", "",
             "| model | split | sigma_vm | displacement | u_x | u_y | peak-stress err |",
             "|---|---|---|---|---|---|---|"]
    for k in acc:
        for key, label in (("in_dist", "in-distribution"), ("ood", "OOD geometry")):
            s = acc[k][key]
            lines.append(
                f"| {k.upper()} | {label} | {s['sigma_vm'] * 100:.2f} "
                f"| {s['displacement'] * 100:.2f} | {s['u_x'] * 100:.2f} "
                f"| {s['u_y'] * 100:.2f} | {s['peak_stress_error'] * 100:.2f} |")
    t = results["timing"]
    lines += ["", "## Inference cost & speedup", "",
              "| pipeline | mode | time per field (ms) | speedup vs FEM |",
              "|---|---|---|---|",
              f"| FEM solve | one solve | {t['fem_solve']['mean_ms']:.2f} "
              f"± {t['fem_solve']['std_ms']:.2f} | 1x |"]
    modes = {"latency_cpu": ("single forward", "CPU"),
             "latency_gpu": ("single forward", "GPU"),
             "throughput_gpu": ("batched throughput", "GPU")}
    for k in acc:
        for key, (mode, hw) in modes.items():
            entry = t.get(k, {}).get(key)
            if entry is not None:
                sp = results["speedup"][f"{k}_{key}"]
                lines.append(f"| {k.upper()} ({hw}) | {mode} | "
                             f"{entry['mean_ms']:.3f} ± {entry['std_ms']:.3f} "
                             f"| {sp:,.0f}x |")
    lines += ["", f"Best speedup: **{results['targets']['best_speedup']:,.0f}x**  ",
              f"OOD sigma_vm rel L2 under 8%: "
              f"**{results['targets']['ood_under_8pct']}**", ""]
    Path(path).write_text("\n".join(lines))


def run_benchmark(checkpoints: dict, dataset_path, data_cfg, out_dir,
                  n_timing_runs: int = 100, batch_size: int = 64,
                  device=None) -> dict:
    """Evaluate every checkpoint vs FEM and write the ``benchmark/`` outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(dataset_path)
    splits = make_splits(raw["theta"], data_cfg)
    cpu = torch.device("cpu")
    accel = get_device(device or "auto")

    models, accuracy = {}, {}
    for path in checkpoints.values():
        model, norm, kind = load_trained(path, device=cpu)
        models[kind] = (model, norm)
        logger.info("evaluating %s on test (%d) and OOD (%d) splits",
                    kind, len(splits.test), len(splits.ood))
        accuracy[kind] = {
            "in_dist": evaluate_split(model, kind, raw, splits.test, norm, cpu,
                                      desc=f"{kind} in-dist"),
            "ood": evaluate_split(model, kind, raw, splits.ood, norm, cpu,
                                  desc=f"{kind} OOD"),
        }

    rep = int(splits.test[len(splits.test) // 2])     # representative sample
    logger.info("timing FEM solve (%d runs)...", n_timing_runs)
    timing = {"fem_solve": time_fem_solve(data_cfg, raw["theta"][rep], n_timing_runs)}
    fem_ms = timing["fem_solve"]["mean_ms"]
    tput_runs = max(n_timing_runs // 5, 5)
    speedup = {}
    for kind, (model, norm) in models.items():
        args = _surrogate_args(kind, raw, rep, norm)
        timing[kind] = {"latency_cpu": time_surrogate(model, kind, args,
                                                      n_timing_runs, cpu)}
        acc_dev = _accel_for(kind, accel)
        if acc_dev is not None:
            # single-forward latency and batched throughput on the accelerator
            timing[kind]["latency_gpu"] = time_surrogate(
                model, kind, args, n_timing_runs, acc_dev)
            timing[kind]["throughput_gpu"] = time_surrogate_throughput(
                model, kind, args, batch_size, tput_runs, acc_dev)
        model.to(cpu)                            # restore for later plotting
        for key, entry in timing[kind].items():
            speedup[f"{kind}_{key}"] = fem_ms / entry["mean_ms"]

    best = min(accuracy, key=lambda k: accuracy[k]["ood"]["sigma_vm"])
    results = {
        "accuracy": accuracy,
        "timing": timing,
        "speedup": speedup,
        "device": {"cpu": str(cpu), "accel": str(accel)},
        "targets": {
            "best_model": best,
            "best_speedup": max(speedup.values()) if speedup else 0.0,
            "speedup_over_1000x": (max(speedup.values()) if speedup else 0.0) >= 1000,
            "ood_under_8pct": any(a["ood"]["sigma_vm"] < 0.08
                                  for a in accuracy.values()),
        },
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    _write_markdown(results, out_dir / "results.md")
    plot_accuracy(results, out_dir / "accuracy.png")
    plot_speedup(results, out_dir / "speedup.png")
    step = max(len(splits.ood) // 3, 1)
    trip_idx = [int(i) for i in splits.ood[::step][:3]]
    bmodel, bnorm = models[best]
    bmodel.to(cpu)                       # timing may have left it on an accelerator
    plot_triptych(bmodel, best, raw, trip_idx, bnorm,
                  out_dir / "triptych.png", cpu)

    logger.info("best model: %s | best speedup: %.0fx | OOD<8%%: %s",
                best, results["targets"]["best_speedup"],
                results["targets"]["ood_under_8pct"])
    for kind in accuracy:
        a = accuracy[kind]
        logger.info("  %s  sigma_vm rel L2  in-dist=%.2f%%  OOD=%.2f%%",
                    kind, a["in_dist"]["sigma_vm"] * 100,
                    a["ood"]["sigma_vm"] * 100)
    return results
