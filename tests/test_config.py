from pathlib import Path
import pytest
from fieldnet.config import DataConfig, FNOConfig, DeepONetConfig, load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_data_config_loads():
    cfg = load_config(CONFIGS / "data.yaml", DataConfig)
    assert cfg.grid_resolution == 128
    assert cfg.n_samples == 2000
    assert cfg.ood_r_threshold == 0.25


def test_fno_config_loads():
    cfg = load_config(CONFIGS / "fno.yaml", FNOConfig)
    assert cfg.modes == 24
    assert cfg.width == 64


def test_deeponet_config_loads():
    cfg = load_config(CONFIGS / "deeponet.yaml", DeepONetConfig)
    assert cfg.n_basis == 128
    assert len(cfg.branch_layers) == 5


def test_data_config_split_fractions():
    cfg = DataConfig()
    assert abs(cfg.train_fraction + cfg.val_fraction + cfg.test_fraction - 1.0) < 1e-6


def test_data_config_invalid_fractions():
    with pytest.raises(Exception):
        DataConfig(train_fraction=0.5, val_fraction=0.5, test_fraction=0.5)
