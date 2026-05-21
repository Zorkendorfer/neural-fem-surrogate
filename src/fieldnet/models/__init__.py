"""Neural operator models: FNO (spectral conv from scratch) and DeepONet."""
from fieldnet.models.deeponet import DeepONet
from fieldnet.models.fno import FNO2d, SpectralConv2d

__all__ = ["DeepONet", "FNO2d", "SpectralConv2d"]
