"""Phase 1 acceptance tests: FEM data factory + Kirsch sanity check."""
import numpy as np

from fieldnet.config import DataConfig
from fieldnet.fem.generate import generate_dataset, sample_theta
from fieldnet.fem.interpolate import compute_sdf_mask, interpolate_to_grid
from fieldnet.fem.kirsch import kirsch_von_mises
from fieldnet.fem.mesh import plate_with_hole_mesh
from fieldnet.fem.solver import solve_plate

CFG = DataConfig()


def test_mesh_geometry():
    r = 0.2
    mesh = plate_with_hole_mesh(r, n_radial=24, n_angular=80, half_width=1.0)
    p = mesh.p
    radius = np.sqrt(p[0] ** 2 + p[1] ** 2)
    assert np.isclose(radius.min(), r, atol=1e-9)         # inner ring on the hole
    assert np.isclose(np.abs(p).max(), 1.0, atol=1e-9)    # outer ring on the square
    assert p.shape[1] == 25 * 80                          # (n_radial+1) * n_angular

    # no degenerate triangles (skfem sorts node indices, so winding is not
    # preserved on mesh.t -- check absolute area instead)
    x, y, t = p[0], p[1], mesh.t
    area = 0.5 * ((x[t[1]] - x[t[0]]) * (y[t[2]] - y[t[0]])
                  - (x[t[2]] - x[t[0]]) * (y[t[1]] - y[t[0]]))
    assert np.all(np.abs(area) > 1e-10)


def test_sample_theta_reproducible_and_in_range():
    a = sample_theta(CFG, 64)
    b = sample_theta(CFG, 64)
    assert np.array_equal(a, b)                           # fixed seed -> identical
    assert a.shape == (64, 3)
    assert (a[:, 0] >= CFG.r_range[0]).all() and (a[:, 0] <= CFG.r_range[1]).all()
    assert (a[:, 1] >= CFG.sigma_inf_range[0]).all() and (a[:, 1] <= CFG.sigma_inf_range[1]).all()
    assert (a[:, 2] >= CFG.alpha_range[0]).all() and (a[:, 2] <= CFG.alpha_range[1]).all()


def test_sdf_mask():
    r = 0.15
    sdf, mask = compute_sdf_mask(r, 128, 1.0)
    assert sdf.shape == (128, 128) and mask.shape == (128, 128)
    assert np.array_equal(mask, sdf > 0.0)
    assert not mask[64, 64]                               # grid centre is in the hole
    assert mask[0, 0]                                     # corner is material


def test_solver_runs():
    mesh = plate_with_hole_mesh(0.15, n_radial=32, n_angular=120)
    sol = solve_plate(mesh, sigma_inf=5.0, alpha_deg=30.0, E=70000.0, nu=0.33)
    assert sol.u.shape == (mesh.p.shape[1], 2)
    assert np.isfinite(sol.u).all()
    assert np.isfinite(sol.nodal_vm).all()
    assert (sol.nodal_vm >= 0.0).all()


def test_kirsch_check():
    """Acceptance: FEM stress matches the Kirsch solution to <5% near the hole."""
    r, sigma_inf = 0.1, 1.0       # small hole -> finite-plate correction negligible
    mesh = plate_with_hole_mesh(r, CFG.mesh_n_radial, CFG.mesh_n_angular,
                                CFG.half_width, CFG.mesh_grading)
    sol = solve_plate(mesh, sigma_inf, alpha_deg=0.0, E=70000.0, nu=0.33)

    cx, cy = sol.centroids[:, 0], sol.centroids[:, 1]
    rho = np.sqrt(cx ** 2 + cy ** 2)
    near = rho < 1.4 * r                                  # element ring around the hole
    fem_vm = sol.elem_vm[near]
    kirsch_vm = kirsch_von_mises(cx[near], cy[near], r, sigma_inf, 0.0)

    # peak stress-concentration factor (K_t = 3)
    peak_err = abs(fem_vm.max() - 3.0 * sigma_inf) / (3.0 * sigma_inf)
    assert peak_err < 0.05, f"peak K_t error {peak_err:.3%}"

    # field agreement near the hole edge
    rel_l2 = np.linalg.norm(fem_vm - kirsch_vm) / np.linalg.norm(kirsch_vm)
    assert rel_l2 < 0.05, f"near-hole rel L2 {rel_l2:.3%}"


def test_interpolate_to_grid():
    mesh = plate_with_hole_mesh(0.2, n_radial=32, n_angular=120)
    sol = solve_plate(mesh, sigma_inf=5.0, alpha_deg=0.0, E=70000.0, nu=0.33)
    nodal = np.column_stack([sol.u[:, 0], sol.u[:, 1], sol.nodal_vm])
    fields, mask = interpolate_to_grid(mesh.p.T, nodal, 0.2, 128, 1.0)
    assert fields.shape == (3, 128, 128)
    assert np.all(fields[:, ~mask] == 0.0)                # hole interior is zeroed
    assert np.isfinite(fields).all()


def test_generate_small(tmp_path):
    """End-to-end: a small dataset writes to disk with the expected layout."""
    out = generate_dataset(CFG, n_samples=4, workers=1, out_path=tmp_path / "ds.npz")
    data = np.load(out)
    assert data["theta"].shape == (4, 3)
    assert data["fields"].shape == (4, 3, 128, 128)
    assert data["sdf"].shape == (4, 128, 128)
    assert data["mask"].shape == (4, 128, 128)
    assert np.isfinite(data["fields"]).all()
    assert list(data["field_names"]) == ["u_x", "u_y", "sigma_vm"]
