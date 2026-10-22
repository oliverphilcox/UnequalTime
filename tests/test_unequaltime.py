import numpy as np
import pytest
import healpy as hp
import unequaltime as ut
from unequaltime import estimator


def test_nufft():
    rng = np.random.default_rng(0)
    t, p = rng.uniform(0, 1, 1000), rng.integers(0, 2, 1000)
    exact = [[np.exp(-1j * np.pi * m * t[p == q]).sum() for m in range(9)] for q in range(2)]
    assert np.abs(estimator.nufft(p, t, 9, 1.) - exact).max() < 1e-5 * len(t)


def test_bin_omega():
    Z = np.random.default_rng(1).normal(size=(1, 5, 5, 3)) + 0j
    X = estimator.pair_vector(*estimator.bin_omega(Z, Z, pairs=True))
    assert X.shape == (1, len(estimator.rows(5)), 3) and np.allclose(X[0, 0], Z[0, 0, 0].real)
    assert np.allclose(estimator.bin_omega(Z, Z, width=2), estimator.omega_bins(5, 2) @ np.einsum('bjjl->bjl', Z).real)


def test_ric_full_sky():
    win = estimator.window(np.ones(12 * 8**2), np.array([0, 4, 8]))
    assert abs(win['noise_shape'][0]) < 1e-12 and np.abs(win['coupling'][0]).max() < 1e-12
    assert np.abs(win['coupling'][1:, 1:] - np.eye(23)).max() < 1e-3


def test_shot_noise_null():
    """Unclustered (Poisson) galaxies in a varying mask give Z consistent with zero, for both background options."""
    nside, rng = 16, np.random.default_rng(2)
    mask = np.clip(1.2 - 0.3 * hp.pix2vec(nside, np.arange(12 * nside**2))[2]**2 - rng.uniform(0, 0.4, 12 * nside**2), 0, 1)
    mask[:len(mask) // 4] = 0
    for nbar in ('data', 4e4):
        survey = ut.Survey(mask, [0.2, 0.5], 4, [2, 12, 24, 40], nbar=nbar)
        Z = []
        for _ in range(20):
            pix = np.repeat(np.arange(len(mask)), rng.poisson(4e4 * 0.3 * mask * hp.nside2pixarea(nside)))
            ra, dec = hp.pix2ang(nside, pix, lonlat=True)
            Z.append(survey.measure(ra, dec, rng.uniform(0.2, 0.5, len(pix))))
        Z = np.array(Z)
        pull = Z.mean(0) / (Z.std(0, ddof=1) / np.sqrt(len(Z)))
        assert np.abs(pull.mean()) < 3 / np.sqrt(pull.size) + 0.1 and np.sqrt(np.mean(pull**2)) < 1.5


def test_pairs_diagonal():
    """With width 1, the Z(A, +A) rows of the pairs data vector, expectation and covariance equal those of Z_ell(omega)."""
    nside, nomega = 8, 3
    survey = ut.Survey(np.ones(12 * nside**2), [0.2, 0.5], nomega, [2, 10, 20], nbar=1e4)
    rng = np.random.default_rng(3)
    G = rng.normal(size=(1, nomega, 1, nomega, 3 * nside)) * 1e-3
    G = G + G.transpose(2, 3, 0, 1, 4) + 0j
    spectra = dict(Z_plus=G, Z_minus=G, nbar=survey.nbar, cross_bins=True)
    pairs = survey.rebin(pairs=True)
    r = pairs.rows
    diag = np.flatnonzero((r[:, 0] == 0) & (r[:, 1] == r[:, 2]))
    idx = (diag[:, None] * 2 + np.arange(2)).ravel()
    assert np.allclose(pairs.vector(pairs.expected(spectra))[idx], survey.vector(survey.expected(spectra)))
    assert np.allclose(pairs.covariance(spectra)[np.ix_(idx, idx)], survey.covariance(spectra))


def test_background():
    """The background estimated from the data counts every galaxy in the bins and the mask."""
    nside, rng = 8, np.random.default_rng(4)
    mask = np.where(np.arange(12 * nside**2) < 300, 0, rng.uniform(0.5, 1, 12 * nside**2))
    pix = rng.integers(0, len(mask), 5000)
    ra, dec = hp.pix2ang(nside, pix, lonlat=True)
    z = rng.uniform(0.1, 0.8, len(pix))
    survey = ut.Survey(mask, [0.2, 0.4, 0.6], 3, [2, 10])
    survey.measure(ra, dec, z)
    zz = np.linspace(0.1, 0.8, 700001)
    ngal = np.sum((mask[pix] > 0) & (z >= 0.2) & (z < 0.6))
    assert abs(np.trapezoid(survey.nbar(zz), zz) * mask.sum() * hp.nside2pixarea(nside) / ngal - 1) < 1e-3


def test_survey_api():
    """Shapes and consistency of the Survey interface: binning, pairs, vector, covariance, rebin and the error cases."""
    nside, nomega, rng = 8, 5, np.random.default_rng(5)
    mask = np.ones(12 * nside**2)
    survey = ut.Survey(mask, [0.2, 0.4, 0.6], nomega, [2, 10, 20])
    with pytest.raises(ValueError):
        survey.theory(None, 1.)                                   # nbar='data' needs a measurement first
    with pytest.raises(ValueError):
        ut.Survey(mask, [0.2, 0.4], nomega, [2, 10], nbar='model')
    with pytest.raises(ValueError):
        survey.rebin(width=0)
    pix = rng.integers(0, len(mask), 20000)
    ra, dec = hp.pix2ang(nside, pix, lonlat=True)
    z = rng.uniform(0.2, 0.6, len(pix))
    pairs = survey.rebin(width=2, pairs=True)                     # made before measure(): shares the estimated nbar
    assert survey.measure(ra, dec, z).shape == (2, nomega, 2) and pairs.nbar is survey.nbar
    Z_plus, Z_minus = pairs.measure(ra, dec, z)
    assert Z_plus.shape == Z_minus.shape == (2, 3, 3, 2) and np.allclose(Z_plus.imag[:, [0, 1, 2], [0, 1, 2]], 0)
    assert pairs.omega.shape == (2, 3) and len(pairs.vector((Z_plus, Z_minus))) == 2 * len(pairs.rows) * 2
    G = rng.normal(size=(2, nomega, 2, nomega, 3 * nside)) * 1e-3
    G = G + G.transpose(2, 3, 0, 1, 4) + 0j
    spectra = dict(Z_plus=G, Z_minus=G, nbar=survey.nbar, cross_bins=True)
    for s in (survey, pairs, survey.rebin(width=3)):
        assert s.covariance(spectra).shape == (len(s.vector(s.expected(spectra))),) * 2
    with pytest.raises(ValueError):
        survey.covariance(dict(spectra, cross_bins=False))


def test_theory():
    """The theory from a CLASS instance: Hermitian, real on the diagonal, and the cross-bin switch."""
    classy = pytest.importorskip('classy')
    cl = classy.Class()
    cl.set({'h': 0.68, 'omega_b': 0.0224, 'omega_cdm': 0.12, 'output': 'mPk', 'P_k_max_1/Mpc': 1.})
    cl.compute()
    survey = ut.Survey(np.ones(12 * 4**2), [0.3, 0.5, 0.7], 3, [2, 6, 12], nbar=1e4)
    full, same = survey.theory(cl, 1.5, kmax=0.05), survey.theory(cl, 1.5, kmax=0.05, cross_bins=False)
    Z = full['Z_plus']
    assert np.allclose(Z, Z.transpose(2, 3, 0, 1, 4).conj()) and np.allclose(np.einsum('pipil->pil', Z).imag, 0)
    assert np.allclose(same['Z_plus'][[0, 1], :, [0, 1]], Z[[0, 1], :, [0, 1]]) and not same['Z_plus'][0, :, 1].any()
