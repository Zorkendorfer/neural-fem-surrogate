"""The three headline benchmark figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fieldnet.eval.inference import predict_grid  # noqa: E402


def plot_accuracy(results: dict, path) -> None:
    """Grouped bar chart: sigma_vm relative L2, in-distribution vs OOD."""
    acc = results["accuracy"]
    models = list(acc)
    x = np.arange(len(models))
    indist = [acc[m]["in_dist"]["sigma_vm"] * 100 for m in models]
    ood = [acc[m]["ood"]["sigma_vm"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, indist, 0.4, label="in-distribution", color="#4c72b0")
    ax.bar(x + 0.2, ood, 0.4, label="OOD geometry (r > 0.25)", color="#dd8452")
    ax.axhline(8.0, ls="--", c="grey", lw=1)
    ax.text(len(models) - 0.5, 8.3, "OOD target 8%", fontsize=8, color="grey")
    for xi, (a, b) in enumerate(zip(indist, ood)):
        ax.text(xi - 0.2, a, f"{a:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, b, f"{b:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [m.upper() for m in models])
    ax.set_ylabel("von Mises stress  relative L2  (%)")
    ax.set_title("Surrogate accuracy vs FEM ground truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_speedup(results: dict, path) -> None:
    """Log-scale per-field cost: FEM vs surrogate (single forward + batched)."""
    timing = results["timing"]
    speedup = results["speedup"]
    nice = {"latency_cpu": "single forward\n(CPU)",
            "latency_gpu": "single forward\n(GPU)",
            "throughput_gpu": "batched\nthroughput (GPU)"}
    color = {"latency_cpu": "#4c72b0", "latency_gpu": "#5a9bd4",
             "throughput_gpu": "#55a868"}

    rows = [("FEM solve", timing["fem_solve"]["mean_ms"], None, "#c44e52")]
    for kind in results["accuracy"]:
        for key in ("latency_cpu", "latency_gpu", "throughput_gpu"):
            entry = timing.get(kind, {}).get(key)
            if entry is not None:
                rows.append((f"{kind.upper()}\n{nice[key]}", entry["mean_ms"],
                             speedup[f"{kind}_{key}"], color[key]))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(rows)), 4.8))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                  color=[r[3] for r in rows])
    ax.set_yscale("log")
    ax.set_ylabel("time per field  (ms, log scale)")
    ax.set_title("Inference cost: FEM vs neural-operator surrogate")
    for bar, (_, val, sp, _c) in zip(bars, rows):
        txt = f"{val:.3g} ms" + (f"\n{sp:,.0f}x" if sp is not None else "")
        ax.text(bar.get_x() + bar.get_width() / 2, val, txt,
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_triptych(model, kind, raw, indices, norm, path, device="cpu") -> None:
    """Rows of samples; columns: FEM truth | surrogate | absolute error (sigma_vm)."""
    hw = float(raw["half_width"])
    extent = [-hw, hw, -hw, hw]
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), squeeze=False)
    for row, idx in enumerate(indices):
        mask = raw["mask"][idx].astype(bool)
        true = np.where(mask, raw["fields"][idx, 2], np.nan)
        pred_full = predict_grid(model, kind, raw, idx, norm, device)
        pred = np.where(mask, pred_full[2], np.nan)
        err = np.abs(pred - true)
        vmax = np.nanmax(true)
        panels = [(true, "FEM truth", 0.0, vmax, "viridis"),
                  (pred, f"{kind.upper()} surrogate", 0.0, vmax, "viridis"),
                  (err, "absolute error", 0.0, np.nanmax(err), "magma")]
        r = raw["theta"][idx]
        for col, (data, title, lo, hi, cmap) in enumerate(panels):
            ax = axes[row][col]
            im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                           vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"r={r[0]:.3f}  sigma={r[1]:.1f}  alpha={r[2]:.0f}",
                              fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("von Mises stress: ground truth vs surrogate (OOD geometries)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
