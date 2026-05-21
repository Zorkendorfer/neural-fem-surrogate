"""Shared test fixtures."""
import numpy as np
import pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    """Synthetic dataset (no FEM) spanning both sides of the OOD threshold.

    Matches the real dataset.npz schema so the data pipeline and trainer can
    be exercised without the 373 MB FEM dataset.
    """
    rng = np.random.default_rng(0)
    n, H = 60, 16
    r = np.concatenate([rng.uniform(0.05, 0.25, 40),
                        rng.uniform(0.2501, 0.30, 20)])
    rng.shuffle(r)
    theta = np.stack([r, rng.uniform(1.0, 10.0, n),
                      rng.uniform(0.0, 90.0, n)], axis=1).astype(np.float32)

    axis = np.linspace(-1.0, 1.0, H)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    rad = np.sqrt(X**2 + Y**2)
    sdf = np.stack([rad - ri for ri in r]).astype(np.float32)
    mask = (sdf > 0).astype(np.uint8)
    fields = rng.normal(size=(n, 3, H, H)).astype(np.float32) * mask[:, None]

    path = tmp_path / "tiny.npz"
    np.savez(path, theta=theta, theta_names=np.array(["r", "sigma_inf", "alpha"]),
             fields=fields, field_names=np.array(["u_x", "u_y", "sigma_vm"]),
             sdf=sdf, mask=mask, coords=axis.astype(np.float32),
             half_width=np.float32(1.0), grid_resolution=np.int64(H),
             seed=np.int64(0))
    return path
