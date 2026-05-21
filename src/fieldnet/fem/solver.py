"""Plane-stress linear elasticity solve for a plate under far-field tension."""
from dataclasses import dataclass

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, condense, solve)
from skfem.helpers import sym_grad


@dataclass
class ElasticitySolution:
    mesh: MeshTri
    u: np.ndarray             # (N, 2) nodal displacement [u_x, u_y]
    nodal_stress: np.ndarray  # (N, 3) [s_xx, s_yy, s_xy] (nodal-averaged)
    nodal_vm: np.ndarray      # (N,)   von Mises stress at nodes
    elem_stress: np.ndarray   # (M, 3) constant per-element stress
    elem_vm: np.ndarray       # (M,)   von Mises stress per element
    centroids: np.ndarray     # (M, 2) element centroids


def _von_mises(stress: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises from a (..., 3) [s_xx, s_yy, s_xy] array."""
    sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def _element_stress(p: np.ndarray, t: np.ndarray, u: np.ndarray,
                    E: float, nu: float):
    """Constant strain/stress per P1 triangle. Returns (M, 3) stress, (M, 2) centroids."""
    x, y = p[0], p[1]
    x0, x1, x2 = x[t[0]], x[t[1]], x[t[2]]
    y0, y1, y2 = y[t[0]], y[t[1]], y[t[2]]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)   # 2 * signed area
    b = np.stack([y1 - y2, y2 - y0, y0 - y1]) / det        # (3, M) dN/dx
    c = np.stack([x2 - x1, x0 - x2, x1 - x0]) / det        # (3, M) dN/dy

    ux, uy = u[t, 0], u[t, 1]                              # (3, M)
    exx = np.sum(ux * b, axis=0)
    eyy = np.sum(uy * c, axis=0)
    exy = 0.5 * (np.sum(ux * c, axis=0) + np.sum(uy * b, axis=0))

    coef = E / (1.0 - nu * nu)
    sxx = coef * (exx + nu * eyy)
    syy = coef * (eyy + nu * exx)
    sxy = (E / (1.0 + nu)) * exy
    stress = np.stack([sxx, syy, sxy], axis=1)
    centroids = np.stack([(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0], axis=1)
    return stress, centroids


def _nodal_average(t: np.ndarray, elem_val: np.ndarray, n_nodes: int) -> np.ndarray:
    """Average a per-element quantity onto nodes (simple unweighted patch average)."""
    acc = np.zeros((n_nodes, elem_val.shape[1]))
    cnt = np.zeros(n_nodes)
    for k in range(t.shape[0]):
        np.add.at(acc, t[k], elem_val)
        np.add.at(cnt, t[k], 1.0)
    return acc / cnt[:, None]


def solve_plate(mesh: MeshTri, sigma_inf: float, alpha_deg: float,
                E: float, nu: float) -> ElasticitySolution:
    """Solve plane-stress elasticity: far-field uniaxial tension ``sigma_inf`` at
    angle ``alpha_deg`` applied as traction on the outer boundary; the hole is
    traction-free. Rigid-body modes are removed by pinning two corner nodes.
    """
    hw = float(np.abs(mesh.p).max())
    element = ElementVector(ElementTriP1())
    basis = Basis(mesh, element)

    lam = E * nu / (1.0 - nu * nu)        # plane-stress effective Lame lambda
    mu = E / (2.0 * (1.0 + nu))

    @BilinearForm
    def stiffness(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        tr = eu[0, 0] + eu[1, 1]
        s00 = 2.0 * mu * eu[0, 0] + lam * tr
        s11 = 2.0 * mu * eu[1, 1] + lam * tr
        s01 = 2.0 * mu * eu[0, 1]
        return s00 * ev[0, 0] + s11 * ev[1, 1] + 2.0 * s01 * ev[0, 1]

    a = np.deg2rad(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    sa00 = sigma_inf * ca * ca            # applied stress = sigma_inf * (n outer n)
    sa11 = sigma_inf * sa * sa
    sa01 = sigma_inf * ca * sa

    @LinearForm
    def traction(v, w):
        nx, ny = w.n[0], w.n[1]
        tx = sa00 * nx + sa01 * ny
        ty = sa01 * nx + sa11 * ny
        return tx * v[0] + ty * v[1]

    outer = mesh.facets_satisfying(
        lambda x: np.maximum(np.abs(x[0]), np.abs(x[1])) > hw - 1e-7,
        boundaries_only=True,
    )
    fb = FacetBasis(mesh, element, facets=outer)

    K = stiffness.assemble(basis)
    f = traction.assemble(fb)

    # pin rigid-body modes: both DOFs of one corner node + one DOF of another
    p = mesh.p
    n_a = int(np.argmin((p[0] + hw) ** 2 + (p[1] + hw) ** 2))
    n_b = int(np.argmin((p[0] - hw) ** 2 + (p[1] + hw) ** 2))
    nd = basis.nodal_dofs
    D = np.array([nd[0, n_a], nd[1, n_a], nd[1, n_b]], dtype=np.int64)

    x = solve(*condense(K, f, D=D))
    u = np.stack([x[nd[0]], x[nd[1]]], axis=1)            # (N, 2)

    elem_stress, centroids = _element_stress(mesh.p, mesh.t, u, E, nu)
    nodal_stress = _nodal_average(mesh.t, elem_stress, mesh.p.shape[1])
    return ElasticitySolution(
        mesh=mesh,
        u=u,
        nodal_stress=nodal_stress,
        nodal_vm=_von_mises(nodal_stress),
        elem_stress=elem_stress,
        elem_vm=_von_mises(elem_stress),
        centroids=centroids,
    )
