"""Structured triangular mesh of a square plate with a centered circular hole."""
import numpy as np
from skfem import MeshTri


def _square_boundary_point(theta: np.ndarray, half_width: float) -> np.ndarray:
    """Where the ray at angle ``theta`` from the origin meets the square boundary."""
    c, s = np.cos(theta), np.sin(theta)
    scale = half_width / np.maximum(np.abs(c), np.abs(s))
    return np.stack([scale * c, scale * s], axis=0)  # (2, n_angular)


def plate_with_hole_mesh(
    r: float,
    n_radial: int,
    n_angular: int,
    half_width: float = 1.0,
    grading: float = 1.7,
) -> MeshTri:
    """Mesh the domain ``[-hw, hw]^2`` minus a centered disk of radius ``r``.

    Topologically an annulus: a structured ``(n_radial+1) x n_angular`` grid is
    blended from the hole circle (inner) to the square boundary (outer). Radial
    ``grading > 1`` concentrates elements near the hole where stress peaks.
    """
    if not (0.0 < r < half_width):
        raise ValueError(f"hole radius {r} must lie in (0, {half_width})")

    theta = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
    inner = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=0)  # (2, n_ang)
    outer = _square_boundary_point(theta, half_width)                 # (2, n_ang)

    frac = (np.arange(n_radial + 1) / n_radial) ** grading            # (n_rad+1,)
    s = frac.reshape(-1, 1, 1)                                        # (n_rad+1,1,1)
    pts = (1.0 - s) * inner[None] + s * outer[None]                   # (n_rad+1,2,n_ang)
    p = pts.transpose(1, 0, 2).reshape(2, -1)                         # node id = i*n_ang+j

    i = np.arange(n_radial)[:, None]
    j = np.arange(n_angular)[None, :]
    jp = (j + 1) % n_angular
    a = i * n_angular + j
    b = i * n_angular + jp
    c = (i + 1) * n_angular + jp
    d = (i + 1) * n_angular + j
    t1 = np.stack([a, b, c], axis=0).reshape(3, -1)
    t2 = np.stack([a, c, d], axis=0).reshape(3, -1)
    t = np.concatenate([t1, t2], axis=1).astype(np.int64)

    # MeshTri sorts each element's node indices internally; the solver's stress
    # recovery uses a winding-independent formula, so no orientation fix-up here.
    return MeshTri(p, t)
