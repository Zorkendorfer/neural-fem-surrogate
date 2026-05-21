"""Evaluation: accuracy metrics, FEM-vs-surrogate benchmark, headline figures."""
from fieldnet.eval.benchmark import run_benchmark
from fieldnet.eval.inference import load_trained, predict_grid
from fieldnet.eval.metrics import evaluate_split, peak_stress_error, relative_l2

__all__ = [
    "run_benchmark",
    "load_trained",
    "predict_grid",
    "evaluate_split",
    "peak_stress_error",
    "relative_l2",
]
