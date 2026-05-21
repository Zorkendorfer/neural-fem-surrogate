"""Fourier Neural Operator with a spectral convolution implemented from scratch.

The spectral convolution is the core of the FNO and is built directly on
``torch.fft`` -- no black-box operator library -- which is the point of the
exercise. The ``neuraloperator`` package is used only as an optional
correctness cross-check elsewhere, never imported here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fieldnet.config import FNOConfig


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: a learnable complex linear map on the lowest
    ``modes1 x modes2`` Fourier coefficients; all higher modes are dropped.

    ``rfft2`` keeps only non-negative horizontal frequencies, so two weight
    blocks are needed to cover the positive and negative vertical frequencies
    (the top and bottom mode bands of the spectrum).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                               dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                               dtype=torch.cfloat))

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in, H, W) . (in, out, H, W) -> (B, out, H, W), complex
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        m1 = min(self.modes1, H // 2)              # clamp for small grids
        m2 = min(self.modes2, W // 2 + 1)

        x_ft = torch.fft.rfft2(x)                  # (B, C, H, W//2+1) complex
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))  # (B, out, H, W) real


class FNO2d(nn.Module):
    """Fourier Neural Operator: lift -> Fourier layers -> project.

    Each Fourier layer sums a spectral convolution (global, low-pass) with a
    1x1 convolution (local) and applies a GELU nonlinearity.
    """

    def __init__(self, modes: int, width: int, n_layers: int,
                 in_channels: int, out_channels: int):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers))
        self.proj1 = nn.Conv2d(width, width, kernel_size=1)
        self.proj2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        n = len(self.spectral)
        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise)):
            x = spec(x) + pw(x)
            if i < n - 1:                          # no activation on last layer
                x = F.gelu(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)

    @classmethod
    def from_config(cls, cfg: FNOConfig) -> "FNO2d":
        return cls(cfg.modes, cfg.width, cfg.n_layers,
                   cfg.in_channels, cfg.out_channels)
