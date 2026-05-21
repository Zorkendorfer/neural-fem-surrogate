from pathlib import Path
from typing import Tuple, Type, TypeVar
import yaml
from pydantic import BaseModel, model_validator

T = TypeVar("T", bound=BaseModel)


class DataConfig(BaseModel):
    r_range: Tuple[float, float] = (0.05, 0.30)
    sigma_inf_range: Tuple[float, float] = (1.0, 10.0)
    alpha_range: Tuple[float, float] = (0.0, 90.0)
    grid_resolution: int = 128
    n_samples: int = 2000
    sampler: str = "sobol"
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    ood_r_threshold: float = 0.25
    data_dir: str = "data/"

    # FEM mesh + material (Phase 1 data factory)
    half_width: float = 1.0
    mesh_n_radial: int = 64
    mesh_n_angular: int = 200
    mesh_grading: float = 1.7
    youngs_modulus: float = 70000.0
    poisson_ratio: float = 0.33

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "DataConfig":
        total = self.train_fraction + self.val_fraction + self.test_fraction
        assert abs(total - 1.0) < 1e-6, f"Split fractions must sum to 1, got {total}"
        return self


class FNOConfig(BaseModel):
    modes: int = 24
    width: int = 64
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 3


class DeepONetConfig(BaseModel):
    branch_layers: list[int] = [3, 128, 128, 128, 128]
    trunk_layers: list[int] = [2, 128, 128, 128, 128]
    n_basis: int = 128
    out_channels: int = 3


def load_config(path: str | Path, model_class: Type[T]) -> T:
    with open(path) as f:
        data = yaml.safe_load(f)
    return model_class(**data)
