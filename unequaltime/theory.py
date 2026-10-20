"""Linear theory and Gaussian covariance of Z_ell(omega, omega')."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from .estimator import bands, rows

trap = lambda x: np.r_[0.5, np.ones(len(x) - 2), 0.5] * (x[1] - x[0])


def spectra(cosmo, b, nz, zbins, nw, lmax, ells=None, kmax=0.2, nk=1500, dchi=0.2):
    """G = <a_i a_j*>, Gt = <a_i a_j> (nbin, nw, nbin, nw, lmax + 1), i = (z-bin, omega_m): linear, non-Limber, Kaiser RSD,
    computed at `ells` and interpolated (zero above max(ells)), k < kmax, radial step dchi. cosmo: arrays z, chi, D, f (D(0) = 1),
    k, pk (linear, z = 0; k in 1/chi units); b(z): linear bias; nz(z): mean density n_g = nz(z) Omega(n) per unit z and sr."""
    k = np.linspace(1e-4, kmax, nk)
    wk = np.sqrt(2 / np.pi * k**2 * np.exp(np.interp(np.log(k), np.log(cosmo['k']), np.log(cosmo['pk']))) * trap(k))[:, None]
    c = lambda z, q: np.interp(z, cosmo['z'], cosmo[q])
    ker = []
    for z0, z1 in zip(zbins[:-1], zbins[1:]):
        z = np.linspace(z0, z1, int((c(z1, 'chi') - c(z0, 'chi')) / dchi) + 2)
        n = nz(z)
        e = np.exp(-1j * np.pi * np.outer(np.arange(nw), z - z0) / (z1 - z0)) * trap(z) * n / np.sqrt(trap(z) @ n**2)
        ker.append((np.outer(k, c(z, 'chi')), b(z) * c(z, 'D'), c(z, 'f') * c(z, 'D'), e))
    ells = np.arange(lmax + 1) if ells is None else ells
    x = np.arange(0, kmax * c(zbins[-1], 'chi') + 2e-3, 1e-3)
    G = np.zeros((2, len(zbins) - 1, nw, len(zbins) - 1, nw, len(ells)), complex)
    for il, l in enumerate(ells):
        j = [spherical_jn(max(n, 0), x) for n in (l - 2, l, l + 2)]
        ddj = (l * (l - 1) * j[0] / (2 * l - 1) - (l**2 / (2 * l - 1) + (l + 1)**2 / (2 * l + 3)) * j[1]
               + (l + 1) * (l + 2) * j[2] / (2 * l + 3)) / (2 * l + 1)
        d = wk * np.concatenate([(bD * np.interp(kc, x, j[1]) - fD * np.interp(kc, x, ddj)) @ e.T for kc, bD, fD, e in ker], 1)
        G[..., il] = np.array([d.T @ d.conj(), d.T @ d]).reshape(G.shape[:-1])
    G = interp1d(ells, G, axis=-1, bounds_error=False, fill_value=0)(np.arange(lmax + 1))
    return G[0], G[1]


def mean(G, Gt, win):
    """Expected Zpp, Zpm (nbin, nw, nw, nband)."""
    i = np.arange(len(G))
    return G[i, :, i] @ win['A'].T, Gt[i, :, i] @ win['A'].T


def covariance(G, Gt, nz, zbins, win, width=1):
    """Gaussian covariance of bin_omega(Zpp, Zpm, width)[0].ravel(), incl. cross-bin blocks (improved narrow-kernel approx.)."""
    nb, nw, nell = G.shape[0], G.shape[1], G.shape[-1]
    m1, m2 = np.mean(win['mask']), np.mean(win['mask']**2)
    nB, nl = len(bands(nw, width)), len(win['D'])
    W, noise = np.kron(np.eye(nb), bands(nw, width)), np.zeros((2, nb * nw, nb * nw), complex)
    for b, (z0, z1) in enumerate(zip(zbins[:-1], zbins[1:])):
        z = np.linspace(z0, z1, 10001)
        n = nz(z)
        e = np.exp(-1j * np.pi * np.outer(np.arange(nw), z - z0) / (z1 - z0))
        w = trap(z) * n * m1 / (m2 * np.trapezoid(n**2, z))
        noise[:, b * nw:(b + 1) * nw, b * nw:(b + 1) * nw] = (e * w) @ e.conj().T, (e * w) @ e.T
    ext = lambda g, t: np.block([[g, t], [t.conj(), g.conj()]])
    band = lambda x: W @ x @ W.T
    S = ext(*(band(x.reshape(nb * nw, nb * nw, nell).transpose(2, 0, 1)) for x in (G, Gt)))
    S = (np.tensordot(win['Mt'] * win['b2'], S, 1) / m2 + win['s'][:, None, None] * ext(*map(band, noise))) / m2
    k = win['D'].any(0)
    U = win['D'][:, k] * S[k].transpose(1, 2, 0)[:, :, None]
    V = U @ (win['M2'] / (2 * np.arange(nell) + 1))[np.ix_(k, k)]
    r = rows(nB)
    i = np.concatenate([b * nB + r[:, 1] for b in range(nb)])
    j = np.concatenate([(r[:, 0] * nb + b) * nB + r[:, 2] for b in range(nb)])
    c = np.tile(np.where(r[:, 3], -1j, 1), nb)[:, None, None]
    neg = lambda x: (x + nb * nB) % (2 * nb * nB)
    H = lambda x: x.conj().swapaxes(-1, -2)
    sp, inv = np.unique(np.c_[i, j], axis=0, return_inverse=True)
    cov = np.zeros((len(i), len(i), nl, nl))
    for a, (p, q) in enumerate(sp):
        T1 = V[p, i] @ H(U[q, j]) + V[p, neg(j)] @ H(U[q, neg(i)])
        T2 = V[p, j] @ H(U[q, i]) + V[p, neg(i)] @ H(U[q, neg(j)])
        for x in np.flatnonzero(inv == a):
            cov[x] = 0.5 * np.real(c[x] * c.conj() * T1 + c[x] * c * T2)
    cov = cov.transpose(0, 2, 1, 3).reshape(len(i) * nl, -1)
    return 0.5 * (cov + cov.T)
