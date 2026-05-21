"""Kirsch analytical solution: infinite plate with a hole under uniaxial tension.

Used as a free correctness check for the FEM solver in the small-hole regime,
where the finite-plate correction is negligible. The stress-concentration
factor at the hole edge is K_t = 3 (peak von Mises = 3 * sigma_inf).
"""
import numpy as np


def kirsch_stress_polar(rho, theta, a, sigma_inf):
    """Polar stresses ``(s_rr, s_tt, s_rt)`` of the Kirsch solution.

    Infinite plate, hole radius ``a``, far-field uniaxial tension ``sigma_inf``
    along ``theta = 0``; ``theta`` is measured from the load axis.
    """
    ar2 = (a / rho) ** 2
    ar4 = ar2 * ar2
    c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
    srr = 0.5 * sigma_inf * (1.0 - ar2) \
        + 0.5 * sigma_inf * (1.0 - 4.0 * ar2 + 3.0 * ar4) * c2
    stt = 0.5 * sigma_inf * (1.0 + ar2) \
        - 0.5 * sigma_inf * (1.0 + 3.0 * ar4) * c2
    srt = -0.5 * sigma_inf * (1.0 + 2.0 * ar2 - 3.0 * ar4) * s2
    return srr, stt, srt


def kirsch_stress_cartesian(x, y, a, sigma_inf, alpha_deg=0.0):
    """Cartesian stresses ``(s_xx, s_yy, s_xy)``; load applied at ``alpha_deg``."""
    alpha = np.deg2rad(alpha_deg)
    rho = np.sqrt(x * x + y * y)
    psi = np.arctan2(y, x)                       # global polar angle
    srr, stt, srt = kirsch_stress_polar(rho, psi - alpha, a, sigma_inf)
    c, s = np.cos(psi), np.sin(psi)
    sxx = srr * c * c + stt * s * s - 2.0 * srt * s * c
    syy = srr * s * s + stt * c * c + 2.0 * srt * s * c
    sxy = (srr - stt) * s * c + srt * (c * c - s * s)
    return sxx, syy, sxy


def kirsch_von_mises(x, y, a, sigma_inf, alpha_deg=0.0):
    """Plane-stress von Mises stress of the Kirsch solution."""
    sxx, syy, sxy = kirsch_stress_cartesian(x, y, a, sigma_inf, alpha_deg)
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
