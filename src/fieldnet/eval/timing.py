"""Wall-clock timing: surrogate forward pass vs full FEM solve."""
import time

import numpy as np
import torch
from tqdm import tqdm

from fieldnet.fem.interpolate import interpolate_to_grid
from fieldnet.fem.mesh import plate_with_hole_mesh
from fieldnet.fem.solver import solve_plate


def _stats(times) -> dict:
    a = np.asarray(times) * 1e3                  # seconds -> milliseconds
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std())}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_fem_solve(data_cfg, theta, n_runs: int = 100) -> dict:
    """Time the full FEM pipeline (mesh -> solve -> rasterize) for one sample."""
    r, sigma_inf, alpha = (float(theta[0]), float(theta[1]), float(theta[2]))
    times = []
    for run in tqdm(range(n_runs + 3), desc="timing FEM solve", leave=False):  # 3 warm-up
        t0 = time.perf_counter()
        mesh = plate_with_hole_mesh(r, data_cfg.mesh_n_radial,
                                    data_cfg.mesh_n_angular, data_cfg.half_width,
                                    data_cfg.mesh_grading)
        sol = solve_plate(mesh, sigma_inf, alpha, data_cfg.youngs_modulus,
                          data_cfg.poisson_ratio)
        nodal = np.column_stack([sol.u[:, 0], sol.u[:, 1], sol.nodal_vm])
        interpolate_to_grid(mesh.p.T, nodal, r, data_cfg.grid_resolution,
                            data_cfg.half_width)
        if run >= 3:
            times.append(time.perf_counter() - t0)
    return _stats(times)


@torch.no_grad()
def time_surrogate(model, kind, args, n_runs: int = 100,
                   device: torch.device = torch.device("cpu")) -> dict:
    """Time a single surrogate forward pass; ``args`` is the model's input tuple."""
    model.eval().to(device)
    args = tuple(a.to(device) for a in args)
    for _ in range(5):                           # warm-up
        model(*args)
    _sync(device)
    times = []
    for _ in tqdm(range(n_runs), desc="timing surrogate", leave=False):
        t0 = time.perf_counter()
        model(*args)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return _stats(times)


@torch.no_grad()
def time_surrogate_throughput(model, kind, args, batch_size: int,
                              n_runs: int = 20,
                              device: torch.device = torch.device("cpu")) -> dict:
    """Time a *batched* forward pass; report amortized wall-clock per field.

    Batching parallelizes across samples on an accelerator, so per-field cost
    drops well below the single-forward latency -- this is the throughput a
    deployed surrogate (or bulk dataset generation) actually delivers.
    """
    model.eval().to(device)
    batched = tuple(a.repeat(batch_size, *([1] * (a.ndim - 1))).to(device)
                    for a in args)
    for _ in range(3):                           # warm-up
        model(*batched)
    _sync(device)
    times = []
    for _ in tqdm(range(n_runs), desc="timing throughput", leave=False):
        t0 = time.perf_counter()
        model(*batched)
        _sync(device)
        times.append((time.perf_counter() - t0) / batch_size)   # per field
    return _stats(times)
