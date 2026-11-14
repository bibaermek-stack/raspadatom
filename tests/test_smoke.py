import numpy as np
import torch

from spectra_ai.model import SpectralAttentionNet
from spectra_ai.nist import load_database
from spectra_ai.synth import SyntheticSpectra, make_spectrum, N_POINTS


def test_database_loads():
    db = load_database()
    assert len(db) >= 20
    assert all((g.wl_nm.between(200, 900)).all() for g in db.values())


def test_spectrum_shape_and_labels():
    db = load_database()
    sp = sorted(db)
    s, y, t = make_spectrum(db, sp, np.random.default_rng(0))
    assert s.shape == (N_POINTS,) and 0 <= s.min() and s.max() <= 1.0 + 1e-6
    assert 1 <= y.sum() <= 3 and 0 <= t <= 1


def test_deterministic_dataset():
    ds = SyntheticSpectra(load_database(), 10, seed=3)
    assert torch.equal(ds[4][0], ds[4][0])


def test_model_forward():
    m = SpectralAttentionNet(29)
    logits, t = m(torch.randn(2, 1, N_POINTS))
    assert logits.shape == (2, 29) and t.shape == (2,)
