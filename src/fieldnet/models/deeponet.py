"""DeepONet: a branch/trunk operator network with a dot-product head.

The branch MLP encodes the parameter vector ``theta``; the trunk MLP encodes
query coordinates; their inner product (per output channel) gives the field
value at each query point. The trunk handles the irregular plate-with-hole
domain natively -- it is only ever evaluated at material points.
"""
import torch
import torch.nn as nn

from fieldnet.config import DeepONetConfig


def _mlp(layers: list[int], activation) -> nn.Sequential:
    """Plain MLP with an activation after every linear layer."""
    mods: list[nn.Module] = []
    for a, b in zip(layers[:-1], layers[1:]):
        mods.append(nn.Linear(a, b))
        mods.append(activation())
    return nn.Sequential(*mods)


class DeepONet(nn.Module):
    """Branch (theta) + trunk (coords) operator net with a dot-product head.

    The trunk produces ``n_basis`` shared basis functions; the branch produces
    a separate coefficient vector per output channel. The field at a query
    point is the channel-wise inner product of the two.
    """

    def __init__(self, branch_layers: list[int], trunk_layers: list[int],
                 n_basis: int, out_channels: int, activation=nn.Tanh):
        super().__init__()
        self.n_basis = n_basis
        self.out_channels = out_channels
        self.branch = _mlp(branch_layers, activation)
        self.trunk = _mlp(trunk_layers, activation)
        self.branch_head = nn.Linear(branch_layers[-1], out_channels * n_basis)
        self.trunk_head = nn.Linear(trunk_layers[-1], n_basis)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, theta: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # theta: (B, p_theta)    coords: (B, n_query, 2)
        b = self.branch_head(self.branch(theta))                 # (B, oc*k)
        b = b.view(-1, self.out_channels, self.n_basis)          # (B, oc, k)
        t = self.trunk_head(self.trunk(coords))                  # (B, n_query, k)
        out = torch.einsum("bck,bqk->bqc", b, t)                 # (B, n_query, oc)
        return out + self.bias

    @classmethod
    def from_config(cls, cfg: DeepONetConfig) -> "DeepONet":
        return cls(cfg.branch_layers, cfg.trunk_layers,
                   cfg.n_basis, cfg.out_channels)
