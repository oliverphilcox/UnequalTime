import numpy as np
import healpy as hp
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
    assert abs(win['noise_shape'][0]) < 1e-12 and np.abs(win['coupling'][0]).max() < 1e-12
    assert np.abs(win['coupling'][1:, 1:] - np.eye(23)).max() < 1e-3


def test_shot_noise_null():
    """Unclustered (Poisson) galaxies in a varying mask give Z consistent with zero, for both background options."""
    nside, rng = 16, np.random.default_rng(2)
    mask = np.clip(1.2 - 0.3 * hp.pix2vec(nside, np.arange(12 * nside**2))[2]**2 - rng.uniform(0, 0.4, 12 * nside**2), 0, 1)
    mask[:len(mask) // 4] = 0
    zbins, nomega, edges = np.array([0.2, 0.5]), 4, np.array([2, 12, 24, 40])
    nz = lambda z: np.where((z >= 0.2) & (z <= 0.5), 4e4, 0.)
    for ric in (True, False):
        win = ut.window(mask, edges, ric)
        Z = []
        for _ in range(20):
            pix = np.repeat(np.arange(len(mask)), rng.poisson(4e4 * 0.3 * mask * hp.nside2pixarea(nside)))
            ra, dec = hp.pix2ang(nside, pix, lonlat=True)
            Z.append(ut.measure(ra, dec, rng.uniform(0.2, 0.5, len(pix)), win, zbins, nomega, nz=None if ric else nz))
        Z = np.array(Z)
        pull = Z.mean(0) / (Z.std(0, ddof=1) / np.sqrt(len(Z)))
        assert np.abs(pull.mean()) < 3 / np.sqrt(pull.size) + 0.1 and np.sqrt(np.mean(pull**2)) < 1.5


def test_pairs_diagonal():
    """With width 1, the Z(A, +A) rows of the pairs data vector, mean and covariance equal those of Z_ell(omega)."""
    nside, nomega, zbins = 8, 3, np.array([0.2, 0.5])
    win = ut.window(np.ones(12 * nside**2), np.array([2, 10, 20]))
    rng = np.random.default_rng(3)
    G = rng.normal(size=(1, nomega, 1, nomega, 3 * nside)) * 1e-3
    G = G + G.transpose(2, 3, 0, 1, 4) + 0j
    nz = lambda z: np.full_like(z, 1e4)
    diag = np.flatnonzero((ut.rows(nomega)[:, 0] == 0) & (ut.rows(nomega)[:, 1] == ut.rows(nomega)[:, 2]))
    assert np.allclose(ut.mean(G, G, win, pairs=True)[:, diag], ut.mean(G, G, win))
    idx = (diag[:, None] * 2 + np.arange(2)).ravel()
    assert np.allclose(ut.covariance(G, G, nz, zbins, win, pairs=True)[np.ix_(idx, idx)], ut.covariance(G, G, nz, zbins, win))
