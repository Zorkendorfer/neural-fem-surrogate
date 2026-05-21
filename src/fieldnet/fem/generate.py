"""Generate the ground-truth ``(theta, field)`` dataset via parametric FEM solves."""
import os
import warnings
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from fieldnet.config import DataConfig
from fieldnet.fem.interpolate import compute_sdf_mask, interpolate_to_grid, reference_grid
from fieldnet.fem.mesh import plate_with_hole_mesh
from fieldnet.fem.solver import solve_plate
from fieldnet.utils.logging import get_logger

logger = get_logger("fieldnet.fem.generate")

THETA_NAMES = ["r", "sigma_inf", "alpha"]
FIELD_NAMES = ["u_x", "u_y", "sigma_vm"]


def sample_theta(cfg: DataConfig, n_samples: int) -> np.ndarray:
    """Sobol-sample the parameter space ``(r, sigma_inf, alpha)``; reproducible."""
    lo = [cfg.r_range[0], cfg.sigma_inf_range[0], cfg.alpha_range[0]]
    hi = [cfg.r_range[1], cfg.sigma_inf_range[1], cfg.alpha_range[1]]
    sampler = qmc.Sobol(d=3, scramble=True, seed=cfg.seed)
    with warnings.catch_warnings():       # n_samples need not be a power of two
        warnings.simplefilter("ignore")
        unit = sampler.random(n_samples)
    return qmc.scale(unit, lo, hi).astype(np.float64)


@dataclass
class _Task:
    """One FEM solve, packaged so it survives pickling to worker processes."""
    index: int
    r: float
    sigma_inf: float
    alpha: float
    grid_resolution: int
    half_width: float
    n_radial: int
    n_angular: int
    grading: float
    E: float
    nu: float


def _limit_blas_threads() -> None:
    """Pin BLAS to a single thread so worker processes don't oversubscribe.

    Every FEM solve runs numpy/scipy linear algebra. With multiple worker
    processes, multi-threaded BLAS (OpenBLAS / MKL / macOS Accelerate) spawns
    far more threads than cores and progress stalls. Set before the pool is
    created so spawned workers inherit it via the environment.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _solve_sample(task: _Task):
    """Mesh, solve, and rasterize one sample. Returns grid fields + SDF + mask."""
    mesh = plate_with_hole_mesh(task.r, task.n_radial, task.n_angular,
                                task.half_width, task.grading)
    sol = solve_plate(mesh, task.sigma_inf, task.alpha, task.E, task.nu)
    nodal = np.column_stack([sol.u[:, 0], sol.u[:, 1], sol.nodal_vm])
    fields, mask = interpolate_to_grid(mesh.p.T, nodal, task.r,
                                       task.grid_resolution, task.half_width)
    sdf, _ = compute_sdf_mask(task.r, task.grid_resolution, task.half_width)
    return task.index, fields, sdf, mask.astype(np.uint8)


def generate_dataset(cfg: DataConfig, n_samples: int | None = None,
                     workers: int = 1, out_path: str | Path | None = None) -> Path:
    """Run all FEM solves and write a compressed ``.npz`` dataset to disk."""
    n = n_samples if n_samples is not None else cfg.n_samples
    H = cfg.grid_resolution
    theta = sample_theta(cfg, n)

    tasks = [
        _Task(i, float(theta[i, 0]), float(theta[i, 1]), float(theta[i, 2]),
              H, cfg.half_width, cfg.mesh_n_radial, cfg.mesh_n_angular,
              cfg.mesh_grading, cfg.youngs_modulus, cfg.poisson_ratio)
        for i in range(n)
    ]

    fields = np.zeros((n, 3, H, H), dtype=np.float32)
    sdf = np.zeros((n, H, H), dtype=np.float32)
    mask = np.zeros((n, H, H), dtype=np.uint8)

    logger.info("solving %d FEM samples (workers=%d, grid=%d)", n, workers, H)
    if workers and workers > 1:
        _limit_blas_threads()                    # before the pool is created
        # "spawn" everywhere: identical behavior on Linux/macOS/Windows and
        # avoids fork-vs-threaded-BLAS deadlocks on Linux.
        with get_context("spawn").Pool(workers) as pool:
            for idx, fl, sd, mk in tqdm(
                pool.imap_unordered(_solve_sample, tasks), total=n, desc="FEM"
            ):
                fields[idx], sdf[idx], mask[idx] = fl, sd, mk
    else:
        for task in tqdm(tasks, total=n, desc="FEM"):
            idx, fl, sd, mk = _solve_sample(task)
            fields[idx], sdf[idx], mask[idx] = fl, sd, mk

    coords, _, _ = reference_grid(H, cfg.half_width)
    out_path = Path(out_path) if out_path else Path(cfg.data_dir) / "dataset.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        theta=theta.astype(np.float32),
        theta_names=np.array(THETA_NAMES),
        fields=fields,
        field_names=np.array(FIELD_NAMES),
        sdf=sdf,
        mask=mask,
        coords=coords.astype(np.float32),
        half_width=np.float32(cfg.half_width),
        grid_resolution=np.int64(H),
        seed=np.int64(cfg.seed),
    )
    logger.info("wrote %d samples -> %s (%.1f MB)", n, out_path,
                out_path.stat().st_size / 1e6)
    return out_path
