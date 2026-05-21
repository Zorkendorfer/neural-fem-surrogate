"""Interpolate FEM nodal fields onto a fixed reference grid; SDF/mask channels."""
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def reference_grid(grid_resolution: int, half_width: float = 1.0):
    """Uniform ``grid_resolution^2`` grid over ``[-hw, hw]^2``.

    Returns ``(coords, X, Y)`` where ``coords`` is the 1-D axis and ``X``/``Y``
    are ``(H, W)`` meshes (row = y, col = x).
    """
    coords = np.linspace(-half_width, half_width, grid_resolution)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    return coords, X, Y


def compute_sdf_mask(r: float, grid_resolution: int, half_width: float = 1.0):
    """Signed distance to the hole boundary and the material mask.

    ``sdf`` is positive in the plate, negative inside the hole; ``mask`` is
    ``True`` on material.
    """
    _, X, Y = reference_grid(grid_resolution, half_width)
    sdf = np.sqrt(X**2 + Y**2) - r
    mask = sdf > 0.0
    return sdf.astype(np.float32), mask


def interpolate_to_grid(points: np.ndarray, values: np.ndarray, r: float,
                        grid_resolution: int, half_width: float = 1.0):
    """Linearly interpolate nodal ``values`` (N, k) onto the reference grid.

    Grid cells inside the hole are zeroed. Returns ``(k, H, W)`` float32 fields
    and the ``(H, W)`` material mask.
    """
    _, X, Y = reference_grid(grid_resolution, half_width)
    interp = LinearNDInterpolator(points, values, fill_value=0.0)
    grid = np.nan_to_num(interp(X, Y), nan=0.0)            # (H, W, k)
    _, mask = compute_sdf_mask(r, grid_resolution, half_width)
    grid = grid * mask[:, :, None]
    return grid.transpose(2, 0, 1).astype(np.float32), mask
