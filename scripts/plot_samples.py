"""Render random dataset samples as field plots (Phase 1 acceptance check).

Example:
    python scripts/plot_samples.py --dataset data/dataset.npz --n 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Render random dataset samples")
    parser.add_argument("--dataset", type=Path, default="data/dataset.npz")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default="data/sample_plots")
    args = parser.parse_args()

    data = np.load(args.dataset)
    theta, fields, mask = data["theta"], data["fields"], data["mask"]
    names = [str(n) for n in data["field_names"]]
    hw = float(data["half_width"])

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(theta), size=min(args.n, len(theta)), replace=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in idxs:
        r, sig, alpha = theta[i]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, channel, name in zip(axes, fields[i], names):
            disp = np.where(mask[i] > 0, channel, np.nan)
            im = ax.imshow(disp, origin="lower", extent=[-hw, hw, -hw, hw],
                           cmap="viridis")
            ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"sample {i}:  r={r:.3f}   sigma_inf={sig:.2f} MPa   "
                     f"alpha={alpha:.1f} deg")
        fig.tight_layout()
        path = args.out_dir / f"sample_{int(i):04d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
