"""Linear theory and Gaussian covariance of Z_ell(omega, omega')."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from .estimator import bands, rows

trap = lambda x: np.r_[0.5, np.ones(len(x) - 2), 0.5] * (x[1] - x[0])


def theory(cosmo, b, nz, zbins, nw, lmax, ells=None, kmax=0.2, nk=1500, dchi=0.2):
    """Non-Limber Kaiser G = <a_i a_j*>, Gt = <a_i a_j> for i = (z-bin, omega_m), (nbin, nw, nbin, nw, lmax + 1), computed at
    `ells` and interpolated. cosmo: arrays z, chi, D, f (D(0) = 1), k, pk (linear, z = 0); b(z), nz(z) = dN/dz dOmega."""
    k = np.linspace(1e-4, kmax, nk)
    wk = 2 / np.pi * k**2 * np.exp(np.interp(np.log(k), np.log(cosmo['k']), np.log(cosmo['pk']))) * trap(k)
    ker = []
    for z0, z1 in zip(zbins[:-1], zbins[1:]):
        c0, c1 = np.interp([z0, z1], cosmo['z'], cosmo['chi'])
        z = np.linspace(z0, z1, int((c1 - c0) / dchi) + 2)
        w = trap(z) * nz(z)
        e = np.exp(-1j * np.pi * np.outer(np.arange(nw), z - z0) / (z1 - z0)) * w / np.sqrt(w @ nz(z))
        D, f = (np.interp(z, cosmo['z'], cosmo[x]) for x in ('D', 'f'))
        ker.append((np.outer(k, np.interp(z, cosmo['z'], cosmo['chi'])), b(z) * D, f * D, e))
    ells = np.arange(lmax + 1) if ells is None else ells
    G = np.zeros((2, len(zbins) - 1, nw, len(zbins) - 1, nw, len(ells)), complex)
    for il, l in enumerate(ells):
        d = []
        for x, bD, fD, e in ker:
            j = spherical_jn(l, x)
            d.append(((bD + fD * (1 - l * (l + 1) / x**2)) * j + 2 * fD / x * (l / x * j - spherical_jn(l + 1, x))) @ e.T)
        d = np.concatenate(d, 1) * np.sqrt(wk)[:, None]
        G[..., il] = np.array([d.T @ d.conj(), d.T @ d]).reshape(G.shape[:-1])
    G = interp1d(ells, G, axis=-1, bounds_error=False, fill_value=0)(np.arange(lmax + 1))
    return G[0], G[1]


def mean(G, Gt, win):
    """Expected bandpowers Zpp, Zpm (nbin, nw, nw, nband) of the estimator."""
    i = np.arange(len(G))
    return G[i, :, i] @ win['A'].T, Gt[i, :, i] @ win['A'].T


def covariance(G, Gt, nz, zbins, win, width=1):
    """Gaussian covariance of bin_omega(Zpp, Zpm, width).ravel() (improved narrow-kernel approximation), incl. cross-bin blocks."""
    nb, nw, L = G.shape[0], G.shape[1], G.shape[-1]
    W = np.kron(np.eye(nb), bands(nw, width))
    n, N = 2 * len(W), np.zeros((2, nb * nw, nb * nw), complex)
    for b, (z0, z1) in enumerate(zip(zbins[:-1], zbins[1:])):
        z = np.linspace(z0, z1, 10001)
        e = np.exp(-1j * np.pi * np.outer(np.arange(nw), z - z0) / (z1 - z0))
        w = trap(z) * nz(z) * win['m1'] / (win['m2'] * np.trapezoid(nz(z)**2, z))
        N[:, b * nw:(b + 1) * nw, b * nw:(b + 1) * nw] = (e * w) @ e.conj().T, (e * w) @ e.T
    ext = lambda g, t: np.block([[g, t], [t.conj(), g.conj()]])
    band = lambda x: W @ x @ W.T
    S = ext(*(band(x.reshape(nb * nw, nb * nw, L).transpose(2, 0, 1)) for x in (G, Gt)))
    Sc = (np.tensordot(win['Mt'] * win['b2'], S, 1) / win['m2'] + win['s'][:, None, None] * ext(*map(band, N))) / win['m2']
    k = win['D'].any(0)
    U = win['D'][:, k] * Sc[k].transpose(1, 2, 0)[:, :, None]
    V = U @ (win['M2'] / (2 * np.arange(L) + 1))[np.ix_(k, k)]
    r, nB = rows(len(W) // nb), len(W) // nb
    i = np.concatenate([b * nB + r[:, 1] for b in range(nb)])
    j = np.concatenate([(r[:, 0] * nb + b) * nB + r[:, 2] for b in range(nb)])
    c = np.tile(np.where(r[:, 3], -1j, 1), nb)[:, None, None]
    m = lambda x: (x + n // 2) % n
    H = lambda x: x.conj().swapaxes(-1, -2)
    sp, inv = np.unique(np.c_[i, j], axis=0, return_inverse=True)
    cov = np.zeros((len(i), len(i)) + U.shape[2:3] * 2)
    for a, (p, q) in enumerate(sp):
        T1 = (V[p, i] @ H(U[q, j]) + V[p, m(j)] @ H(U[q, m(i)]))
        T2 = (V[p, j] @ H(U[q, i]) + V[p, m(i)] @ H(U[q, m(j)]))
        for x in np.flatnonzero(inv == a):
            cov[x] = 0.5 * np.real(c[x] * c.conj() * T1 + c[x] * c * T2)
    cov = cov.transpose(0, 2, 1, 3).reshape(len(i) * cov.shape[2], -1)
    return 0.5 * (cov + cov.T)
