"""Phase 3 acceptance tests: FNO and DeepONet forward + single-sample overfit."""
import torch
import torch.nn.functional as F

from fieldnet.config import DeepONetConfig, FNOConfig
from fieldnet.models.deeponet import DeepONet
from fieldnet.models.fno import FNO2d, SpectralConv2d


def _smooth_field(batch: int, channels: int, H: int, seed: int = 1) -> torch.Tensor:
    """Low-frequency field -- representable by a band-limited FNO."""
    ax = torch.linspace(-1.0, 1.0, H)
    X, Y = torch.meshgrid(ax, ax, indexing="xy")
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(batch, channels, H, H)
    for c in range(channels):
        f = torch.zeros(H, H)
        for k in range(1, 4):
            a = torch.rand(1, generator=g).item()
            b = torch.rand(1, generator=g).item()
            f = f + a * torch.sin(k * X) * torch.cos(k * Y) + b * torch.cos(k * X)
        out[:, c] = f
    return out


def _overfit(closure, params, steps: int = 400, lr: float = 3e-3):
    """Run Adam on a single-sample loss closure; return (first, last) loss."""
    opt = torch.optim.Adam(params, lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        loss = closure()
        loss.backward()
        opt.step()
        last = loss.item()
        if step == 0:
            first = last
    return first, last


def test_spectral_conv_preserves_shape():
    sc = SpectralConv2d(8, 8, modes1=6, modes2=6)
    out = sc(torch.randn(2, 8, 32, 32))
    assert out.shape == (2, 8, 32, 32)
    assert out.dtype == torch.float32


def test_fno_forward_shape():
    model = FNO2d.from_config(FNOConfig())
    y = model(torch.randn(2, 4, 64, 64))
    assert y.shape == (2, 3, 64, 64)


def test_deeponet_forward_shape():
    model = DeepONet.from_config(DeepONetConfig())
    out = model(torch.randn(4, 3), torch.randn(4, 100, 2))
    assert out.shape == (4, 100, 3)


def test_fno_overfits_single_sample():
    """Acceptance: the FNO drives one sample to ~0 loss."""
    torch.manual_seed(0)
    H = 24
    model = FNO2d(modes=8, width=32, n_layers=4, in_channels=4, out_channels=3)
    x = torch.randn(1, 4, H, H)
    y = _smooth_field(1, 3, H)
    first, last = _overfit(lambda: F.mse_loss(model(x), y), model.parameters())
    assert last < first * 1e-2
    assert last < 1e-3


def test_deeponet_overfits_single_sample():
    """Acceptance: the DeepONet drives one sample to ~0 loss."""
    torch.manual_seed(0)
    model = DeepONet(branch_layers=[3, 64, 64], trunk_layers=[2, 64, 64],
                     n_basis=64, out_channels=3)
    theta = torch.randn(1, 3)
    coords = torch.rand(1, 96, 2) * 2.0 - 1.0
    cx, cy = coords[..., 0], coords[..., 1]
    target = torch.stack([torch.sin(cx) * torch.cos(cy),
                          cx ** 2 - cy,
                          torch.sin(2 * cx + cy)], dim=-1)         # (1, 96, 3)
    first, last = _overfit(lambda: F.mse_loss(model(theta, coords), target),
                           model.parameters())
    assert last < first * 1e-2
    assert last < 1e-3
