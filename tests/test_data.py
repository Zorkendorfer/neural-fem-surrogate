"""Phase 2 acceptance tests: splits, normalization, and model-view loaders."""
import numpy as np
import pytest
import torch

from fieldnet.config import DataConfig
from fieldnet.data.dataset import build_dataloaders
from fieldnet.data.normalization import Normalizer
from fieldnet.data.splits import make_splits

CFG = DataConfig()


@pytest.fixture
def tiny_dataset(tmp_path):
    """Synthetic dataset (no FEM) spanning both sides of the OOD threshold."""
    rng = np.random.default_rng(0)
    n, H = 60, 16
    r = np.concatenate([rng.uniform(0.05, 0.25, 40),
                        rng.uniform(0.2501, 0.30, 20)])
    rng.shuffle(r)
    theta = np.stack([r, rng.uniform(1.0, 10.0, n),
                      rng.uniform(0.0, 90.0, n)], axis=1).astype(np.float32)

    axis = np.linspace(-1.0, 1.0, H)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    rad = np.sqrt(X**2 + Y**2)
    sdf = np.stack([rad - ri for ri in r]).astype(np.float32)
    mask = (sdf > 0).astype(np.uint8)
    fields = rng.normal(size=(n, 3, H, H)).astype(np.float32) * mask[:, None]

    path = tmp_path / "tiny.npz"
    np.savez(path, theta=theta, theta_names=np.array(["r", "sigma_inf", "alpha"]),
             fields=fields, field_names=np.array(["u_x", "u_y", "sigma_vm"]),
             sdf=sdf, mask=mask, coords=axis.astype(np.float32),
             half_width=np.float32(1.0), grid_resolution=np.int64(H),
             seed=np.int64(0))
    return path


def _raw(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def test_splits_partition_is_disjoint_and_complete():
    theta = np.zeros((100, 3), np.float32)
    theta[:, 0] = np.linspace(0.05, 0.30, 100)
    s = make_splits(theta, CFG)
    allidx = np.concatenate([s.train, s.val, s.test, s.ood])
    assert len(allidx) == 100
    assert len(np.unique(allidx)) == 100               # disjoint + covers all


def test_ood_zero_geometry_overlap(tiny_dataset):
    """Acceptance: the OOD set shares no geometry with train."""
    theta = _raw(tiny_dataset)["theta"]
    s = make_splits(theta, CFG)
    assert set(s.train).isdisjoint(set(s.ood))
    assert (theta[s.ood, 0] > CFG.ood_r_threshold).all()
    assert (theta[s.train, 0] <= CFG.ood_r_threshold).all()
    assert theta[s.ood, 0].min() > theta[s.train, 0].max()   # ranges disjoint


def test_normalizer_roundtrip(tiny_dataset):
    raw = _raw(tiny_dataset)
    s = make_splits(raw["theta"], CFG)
    norm = Normalizer.fit(raw["fields"], raw["sdf"], raw["mask"], s.train, CFG)
    x = raw["fields"][0]
    assert np.allclose(norm.denorm_fields(norm.norm_fields(x)), x, atol=1e-4)
    t = raw["theta"][3]
    assert np.allclose(norm.denorm_theta(norm.norm_theta(t)), t, atol=1e-4)


def test_normalizer_standardizes_train_fields(tiny_dataset):
    raw = _raw(tiny_dataset)
    s = make_splits(raw["theta"], CFG)
    norm = Normalizer.fit(raw["fields"], raw["sdf"], raw["mask"], s.train, CFG)
    nf = np.stack([norm.norm_fields(raw["fields"][i]) for i in s.train])
    m = raw["mask"][s.train].astype(bool)
    for c in range(3):
        vals = nf[:, c][m]
        assert abs(vals.mean()) < 1e-4
        assert abs(vals.std() - 1.0) < 1e-3


def test_normalizer_save_load(tiny_dataset, tmp_path):
    raw = _raw(tiny_dataset)
    s = make_splits(raw["theta"], CFG)
    norm = Normalizer.fit(raw["fields"], raw["sdf"], raw["mask"], s.train, CFG)
    norm.save(tmp_path / "stats.npz")
    loaded = Normalizer.load(tmp_path / "stats.npz")
    assert np.allclose(loaded.field_mean, norm.field_mean)
    assert np.allclose(loaded.field_std, norm.field_std)
    assert loaded.sdf_std == pytest.approx(norm.sdf_std)
    assert np.allclose(loaded.theta_lo, norm.theta_lo)


def test_fno_loader_shapes(tiny_dataset):
    loaders, _, _ = build_dataloaders(tiny_dataset, CFG, view="fno", batch_size=8)
    batch = next(iter(loaders["train"]))
    b = batch["input"].shape[0]
    assert batch["input"].shape == (b, 4, 16, 16)      # [SDF, mask, sigma, alpha]
    assert batch["target"].shape == (b, 3, 16, 16)
    assert batch["mask"].shape == (b, 16, 16)
    assert batch["theta"].shape == (b, 3)
    assert batch["input"].dtype == torch.float32
    # the hole region of the target is zeroed
    holes = batch["mask"].unsqueeze(1).expand_as(batch["target"]) == 0
    assert torch.all(batch["target"][holes] == 0)


def test_deeponet_loader_shapes(tiny_dataset):
    loaders, _, _ = build_dataloaders(tiny_dataset, CFG, view="deeponet",
                                      batch_size=8, n_query=64)
    batch = next(iter(loaders["train"]))
    b = batch["theta"].shape[0]
    assert batch["theta"].shape == (b, 3)
    assert batch["coords"].shape == (b, 64, 2)
    assert batch["target"].shape == (b, 64, 3)
    assert batch["coords"].abs().max() <= 1.0          # query points in-domain


def test_ood_loader_nonempty(tiny_dataset):
    loaders, _, splits = build_dataloaders(tiny_dataset, CFG, view="fno",
                                           batch_size=8)
    assert len(splits.ood) > 0
    assert len(loaders["ood"].dataset) == len(splits.ood)
