"""Benchmark the trained surrogates against the FEM ground truth.

Writes the accuracy table, speedup table, and three headline figures to
``benchmark/``.

Example:
    python scripts/bench.py
    python scripts/bench.py --timing-runs 100 --device cpu
"""
import argparse
from pathlib import Path

from fieldnet.config import DataConfig, load_config
from fieldnet.eval.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark surrogates vs FEM")
    parser.add_argument("--fno", type=Path, default="checkpoints/fno/best.pt")
    parser.add_argument("--deeponet", type=Path,
                        default="checkpoints/deeponet/best.pt")
    parser.add_argument("--dataset", type=Path, default="data/dataset.npz")
    parser.add_argument("--data-config", type=Path, default="configs/data.yaml")
    parser.add_argument("--out-dir", type=Path, default="benchmark")
    parser.add_argument("--timing-runs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for the throughput measurement")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu / auto")
    args = parser.parse_args()

    checkpoints = {}
    for kind, path in (("fno", args.fno), ("deeponet", args.deeponet)):
        if path.exists():
            checkpoints[kind] = path
        else:
            print(f"warning: {kind} checkpoint not found at {path} -- skipping")
    if not checkpoints:
        raise SystemExit("no checkpoints found; train a model first")

    data_cfg = load_config(args.data_config, DataConfig)
    results = run_benchmark(checkpoints, args.dataset, data_cfg, args.out_dir,
                            n_timing_runs=args.timing_runs,
                            batch_size=args.batch_size, device=args.device)
    t = results["targets"]
    print(f"\nbest model: {t['best_model']}  |  best speedup: "
          f"{t['best_speedup']:,.0f}x  |  OOD sigma_vm < 8%: {t['ood_under_8pct']}")
    print(f"outputs -> {args.out_dir}/")


if __name__ == "__main__":
    main()
