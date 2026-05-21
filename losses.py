"""Loss functions for FieldNet, including relative L2 and physics residuals."""
import torch

def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, reduce: bool = True) -> torch.Tensor:
    """Compute relative L2 error: ||pred - target||_2 / ||target||_2.
    
    Args:
        pred: Predicted field.
        target: Ground truth field.
        mask: Binary domain mask. If provided, loss is only computed inside the domain.
        reduce: If True, returns the mean over the batch. If False, returns per-sample loss.
    """
    if mask is not None:
        # FNO case: mask is (B, 1, H, W) or (B, H, W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        pred = pred * mask
        target = target * mask

    # Flatten spatial/channel dimensions: (B, -1)
    diff = (pred - target).reshape(pred.shape[0], -1)
    gold = target.reshape(target.shape[0], -1)

    eps = 1e-8
    # ||pred - target||_2 / ||target||_2
    err = torch.norm(diff, p=2, dim=1) / (torch.norm(gold, p=2, dim=1) + eps)
    
    return err.mean() if reduce else err

def equilibrium_residual(u: torch.Tensor, mask: torch.Tensor, dx: float, E: float, nu: float) -> torch.Tensor:
    """Placeholder for the Phase 8 stretch goal: PDE residual loss.
    
    This would compute the divergence of the stress tensor (div sigma = 0)
    using finite differences (for FNO) or autograd (for DeepONet).
    
    For now, it returns 0 to allow the pipeline to run with physics_loss_weight > 0.
    """
    return torch.tensor(0.0, device=u.device, requires_grad=True)