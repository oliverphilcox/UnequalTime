import numpy as np
import unequaltime as ut
from unequaltime.estimator import nufft


def test_nufft():
    rng = np.random.default_rng(0)
    t, p = rng.uniform(0, 1, 1000), rng.integers(0, 2, 1000)
    exact = [[np.exp(-1j * np.pi * m * t[p == q]).sum() for m in range(9)] for q in range(2)]
    assert np.abs(nufft(p, t, 9, 1.) - exact).max() < 1e-5 * len(t)


def test_bin_omega():
    Z = np.random.default_rng(1).normal(size=(1, 5, 5, 3)) + 0j
    X = ut.bin_omega(Z, Z, pairs=True)
    assert X.shape == (1, len(ut.rows(5)), 3) and np.allclose(X[0, 0], Z[0, 0, 0].real)
    assert np.allclose(ut.bin_omega(Z, Z, width=2), ut.bands(5, 2) @ np.einsum('bjjl->bjl', Z).real)


def test_ric_full_sky():
    win = ut.window(np.ones(12 * 8**2), np.array([0, 4, 8]))
    assert abs(win['s'][0]) < 1e-12 and np.abs(win['Mt'][0]).max() < 1e-12 and np.abs(win['Mt'][1:, 1:] - np.eye(23)).max() < 1e-3
