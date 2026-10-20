"""Z_ell(omega, omega') of a (RA, Dec, z) catalogue with angular selection Omega(n)."""
import numpy as np
import healpy as hp
import pymaster as nmt


def nufft(p, t, nw, L, W=7):
    """sum_g exp(-i omega_m t_g) in each pixel p = 0, 1, ..., omega_m = pi m / L (m < nw), t in [0, L)."""
    es = lambda x: np.exp(2.3 * W * (np.sqrt(np.maximum(1 - x**2, 0)) - 1))
    n = 4 * nw
    u = t * n / (2 * L)
    j = np.floor(u - W / 2).astype(int)[:, None] + 1 + np.arange(W)
    g = np.bincount((p[:, None] * n + j % n).ravel(), es((u[:, None] - j) / (W / 2)).ravel(), (p.max() + 1) * n)
    y = np.linspace(-1, 1, 2001)
    phi = W / 2 * np.trapezoid(es(y) * np.cos(np.pi * W / n * np.outer(np.arange(nw), y)), y)
    return np.fft.rfft(g.reshape(-1, n))[:, :nw] / phi


def window(mask, edges, ric=True):
    """Mask operators for bandpowers [edges[i], edges[i + 1]): theory operator A, deconvolved binning D, coupling Mt of Omega
    (with the radial integral constraint if ric), coupling M2 of Omega^2, shot-noise shape s and pixel window b2."""
    nside, edges = hp.npix2nside(len(mask)), np.asarray(edges)
    l = np.arange(3 * nside)

    def mcm(m):
        f = nmt.NmtField(m, None, spin=0)
        return nmt.NmtWorkspace.from_fields(f, f, nmt.NmtBin.from_lmax_linear(l[-1], 1)).get_coupling_matrix()
    M, alm = mcm(mask), hp.map2alm(mask, iter=3)
    cl, a0 = hp.alm2cl(alm), alm[0].real
    if ric:
        lm = hp.Alm.getlm(l[-1])[0]
        X = np.array([hp.alm2cl(alm, hp.map2alm(mask * hp.alm2map(alm * (lm == L), nside), iter=3)) for L in l]).T
        M = M - 2 * X / (a0 * np.sqrt(4 * np.pi)) + np.outer(cl / a0**2, (2 * l + 1) * cl / (4 * np.pi))
    b2, B = hp.pixwin(nside)**2, (l >= edges[:-1, None]) & (l < edges[1:, None])
    D = B / B.sum(1)[:, None] / b2
    return dict(A=D @ M * b2 / np.mean(mask**2), D=D, Mt=M, M2=mcm(mask**2), s=1 - ric * cl / a0**2, b2=b2, mask=mask, ric=ric)


def measure(ra, dec, z, win, zbins, nw, nz=None, dz=0.005):
    """Zpp = Z(omega_i, omega_j), Zpm = Z(omega_i, -omega_j), (nbin, nw, nw, nband), omega_m = pi m / (bin width), z from the bin
    start. nz(z): mean density n_g = nz(z) Omega(n); None estimates it from the catalogue in shells dz (use window(ric=True))."""
    assert (nz is None) == win['ric'], 'the background is estimated from the catalogue (nz=None) iff window(ric=True)'
    mask = win['mask']
    nside = hp.npix2nside(len(mask))
    area, lmax, pix = hp.nside2pixarea(nside), 3 * nside - 1, hp.ang2pix(nside, ra, dec, lonlat=True)
    i, j = np.indices((nw, nw))
    Z = np.zeros((2, len(zbins) - 1, nw, nw, len(win['D'])), complex)
    for b, (z0, L) in enumerate(zip(zbins[:-1], np.diff(zbins))):
        sel = (z >= z0) & (z < z0 + L) & (mask[pix] > 0)
        t = z[sel] - z0
        u, p = np.unique(pix[sel], return_inverse=True)
        F = np.zeros((nw, len(mask)), complex)
        F[:, u] = nufft(p, t, nw, L).T / area
        if nz is None:
            n = np.histogram(t, max(1, round(L / dz)), (0, L))[0]
            N = np.mean(mask**2) * np.sum(n**2 - n) * len(n) / L / (mask.sum() * area)**2
            F -= np.outer(F.sum(1) / mask.sum(), mask)
        else:
            zz = np.linspace(z0, z0 + L, 10001)
            n = nz(zz)
            N = np.mean(mask**2) * np.trapezoid(n**2, zz)
            F -= np.outer(np.trapezoid(n * np.exp(-1j * np.pi * np.outer(np.arange(nw), zz - z0) / L), zz), mask)
        a = hp.map2alm(np.concatenate([F.real, F.imag]), iter=3, pol=False) / np.sqrt(N)
        P, Q = a[:nw] + 1j * a[nw:], a[:nw] - 1j * a[nw:]
        C = np.zeros((2, nw, nw, lmax + 1), complex)
        for l in range(lmax + 1):
            k = hp.Alm.getidx(lmax, l, np.arange(l + 1))
            x, y = P[:, k], Q[:, k]
            C[0, ..., l] = (x @ x.conj().T + (y[:, 1:] @ y[:, 1:].conj().T).conj()) / (2 * l + 1)
            C[1, ..., l] = (x @ y.conj().T + (y[:, 1:] @ x[:, 1:].conj().T).conj()) / (2 * l + 1)
        S = nufft(0 * p, t, 2 * nw - 1, L)[0] / (4 * np.pi * N)
        noise = np.array([np.where(i >= j, S[abs(i - j)], S[abs(i - j)].conj()), S[i + j]])
        Z[:, b] = (C - noise[..., None] * win['s']) @ win['D'].T
    return Z[0], Z[1]


def bands(nw, width):
    """omega band averaging (nband, nw): band 0 = {omega_0}, then `width` consecutive modes each."""
    k = np.r_[0, np.arange(nw - 1) // width + 1]
    return (k == np.arange(k[-1] + 1)[:, None]) / np.bincount(k)[:, None]


def rows(nB):
    """(kind, A, B, part) of the data vector: Z(A, +B) (kind 0, A <= B), Z(A, -B) (kind 1, 0 < A <= B); Re (0), Im (1)."""
    return np.array([(c, a, b, p) for c in (0, 1) for a in range(c, nB) for b in range(a, nB) for p in (0, 1)
                     if not (c == 0 and a == b and p)])


def bin_omega(Zpp, Zpm, width=1):
    """Data vector of omega-band pairs (nbin, nrow, nband) and its rows."""
    W = bands(Zpp.shape[1], width)
    r = rows(len(W))
    Z = np.einsum('Aa,kzabl,Bb->kzABl', W, np.stack([Zpp, Zpm]), W)[r[:, 0], :, r[:, 1], r[:, 2]]
    return np.where(r[:, 3, None, None], Z.imag, Z.real).swapaxes(0, 1), r
