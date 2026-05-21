"""Cross-platform utility tests."""
import torch

from fieldnet.utils.device import get_device


def test_get_device_auto():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_get_device_explicit_cpu():
    assert get_device("cpu").type == "cpu"
