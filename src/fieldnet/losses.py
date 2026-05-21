"""Training losses: relative L2 field error and an optional physics residual."""
import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None, reduce: bool = True,
                eps: float = 1e-8) -> torch.Tensor:
    """Per-sample relative L2 error ``||pred - target|| / ||target||``.

    ``mask`` (e.g. ``(B, H, W)`` for a ``(B, C, H, W)`` field) restricts the
    norm to material points; it is unsqueezed to broadcast over channels.
    Returns the batch mean if ``reduce``, else a ``(B,)`` tensor.
    """
    if mask is not None:
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(1)
        pred = pred * mask
        target = target * mask
    dims = tuple(range(1, pred.ndim))
    num = ((pred - target) ** 2).sum(dims).clamp_min(1e-12).sqrt()
    den = (target ** 2).sum(dims).clamp_min(eps).sqrt()
    rel = num / den
    return rel.mean() if reduce else rel


def equilibrium_residual(u: torch.Tensor, mask: torch.Tensor, dx: float,
                         E: float, nu: float) -> torch.Tensor:
    """Mean-squared plane-stress equilibrium residual ``div(sigma(u))``.

    ``u`` is a predicted displacement field ``(B, 2, H, W)`` in physical units.
    Stress is recovered from the constitutive law and its divergence is taken
    by central finite differences; the residual is averaged over material
    cells. With no body force, exact elasticity gives a zero residual, so this
    is a soft physics-consistency penalty (gated by a config weight).
    """
    ux, uy = u[:, 0], u[:, 1]                              # (B, H, W)
    dux_dy, dux_dx = torch.gradient(ux, spacing=dx, dim=(1, 2))
    duy_dy, duy_dx = torch.gradient(uy, spacing=dx, dim=(1, 2))

    exx, eyy = dux_dx, duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    coef = E / (1.0 - nu * nu)
    sxx = coef * (exx + nu * eyy)
    syy = coef * (eyy + nu * exx)
    sxy = (E / (1.0 + nu)) * exy

    _, dsxx_dx = torch.gradient(sxx, spacing=dx, dim=(1, 2))
    dsxy_dy, dsxy_dx = torch.gradient(sxy, spacing=dx, dim=(1, 2))
    dsyy_dy, _ = torch.gradient(syy, spacing=dx, dim=(1, 2))

    res = (dsxx_dx + dsxy_dy) ** 2 + (dsxy_dx + dsyy_dy) ** 2
    m = mask.bool()
    return res[m].mean() if m.any() else res.mean() * 0.0
