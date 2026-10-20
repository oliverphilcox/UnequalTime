"""Estimator of the three-dimensional angular power spectra Z_ell(omega, omega') of a (RA, Dec, z) catalog.

In each redshift bin the frequency-space field

    a_lm(omega) = N^{-1/2} int dn Y*_lm(n) int dz exp(-i omega z) [n_g(n, z) - nbar_g(n, z)] w(z)
    N = int dz int dn/4pi nbar_g^2(n, z) w^2(z)

is formed on a HEALPix grid, with omega = m pi / (bin width). The power spectrum
Z_ell(omega, omega') = sum_m a_lm(omega) a*_lm(omega') / (2 ell + 1), minus the shot noise, is binned in ell. The spectra are
coupled pseudo-C_ell: the mask is not deconvolved but applied to the theory, using mode-coupling matrices from NaMaster.
"""
import numpy as np
import healpy as hp
import pymaster as nmt


def nufft(ipix, t, nomega, length, weights=None, support=7):
    """Weighted sum of exp(-i omega_m t) over the points in each pixel, by a non-uniform FFT.

    The points are spread onto a periodic grid of 4 nomega cells on [0, 2 length) with an exponential-of-semicircle kernel,
    Fourier transformed, and divided by the kernel transform (Barnett et al. 2019, arXiv:1808.06736); relative accuracy ~1e-7.

    Parameters
    ----------
    ipix : (npoint,) int array
        Pixel index of each point, in 0, 1, ..., npix - 1.
    t : (npoint,) array
        Positions, in [0, length).
    nomega : int
        Number of frequencies omega_m = pi m / length, m = 0, ..., nomega - 1.
    length : float
        Length of the interval.
    weights : (npoint,) array, optional
        Weight of each point (default 1).
    support : int, optional
        Width of the spreading kernel in grid cells.

    Returns
    -------
    (ipix.max() + 1, nomega) complex array
        Sum over the points in each pixel of weight * exp(-i omega_m t).
    """
    def kernel(x):                                                                         # x in [-1, 1]
        return np.exp(2.3 * support * (np.sqrt(np.maximum(1 - x**2, 0)) - 1))
    ngrid = 4 * nomega
    u = t * ngrid / (2 * length)                                                           # positions in grid cells
    cells = np.floor(u - support / 2).astype(int)[:, None] + 1 + np.arange(support)        # nearest `support` cells
    spread = kernel((u[:, None] - cells) / (support / 2)) * (1 if weights is None else weights[:, None])
    grid = np.bincount((ipix[:, None] * ngrid + cells % ngrid).ravel(), spread.ravel(), (ipix.max() + 1) * ngrid)
    y = np.linspace(-1, 1, 2001)
    kernel_ft = support / 2 * np.trapezoid(kernel(y) * np.cos(np.pi * support / ngrid * np.outer(np.arange(nomega), y)), y)
    return np.fft.rfft(grid.reshape(-1, ngrid))[:, :nomega] / kernel_ft


def window(mask, ell_edges, ric=True):
    """Mask operators for the estimator, its theory and its covariance.

    Parameters
    ----------
    mask : (npix,) array
        HEALPix map (RING ordering) of the angular selection Omega >= 0, in any normalization. Its resolution nside sets the
        multipole range, ell = 0, ..., 3 nside - 1.
    ell_edges : (nband + 1,) int array
        Bandpower edges: band i contains ell_edges[i] <= ell < ell_edges[i + 1], with ell_edges[-1] <= 3 nside.
    ric : bool, optional
        Whether the background n(z) will be estimated from the catalog (measure with nz=None). This imposes a radial integral
        constraint (the masked mean of each map vanishes), which is included in the mode coupling and the shot noise.

    Returns
    -------
    dict
        theory : (nband, 3 nside) array, operator taking full-sky spectra to the expected bandpowers (used by theory.mean);
        binning : (nband, 3 nside) array, binning of coupled spectra with the pixel window removed;
        coupling : (3 nside, 3 nside) array, mode-coupling matrix of the mask (with the integral constraint if ric);
        coupling2 : (3 nside, 3 nside) array, mode-coupling matrix of mask^2 (for the covariance);
        noise_shape : (3 nside,) array, multipole dependence of the shot noise (1 without the integral constraint);
        pixwin2 : (3 nside,) array, pixel window squared;
        mask, ric : the inputs.
    """
    nside, ell_edges = hp.npix2nside(len(mask)), np.asarray(ell_edges)
    ell = np.arange(3 * nside)
    if ell_edges[-1] > len(ell):
        raise ValueError(f'ell_edges must not exceed 3 nside = {len(ell)}')

    def coupling(w):                                                     # NaMaster mode-coupling matrix of the window w
        f = nmt.NmtField(w, None, spin=0)
        return nmt.NmtWorkspace.from_fields(f, f, nmt.NmtBin.from_lmax_linear(ell[-1], 1)).get_coupling_matrix()
    M, mask_lm = coupling(mask), hp.map2alm(mask, iter=3)
    mask_cl, mask_00 = hp.alm2cl(mask_lm), mask_lm[0].real
    if ric:
        # The data-estimated background nulls the masked monopole of each map, a_lm -> a_lm - (Omega_lm / Omega_00) a_00,
        # so the coupling becomes M - 2 X + Y Z^T, with X[l, L] = C_l(Omega, Omega Omega_L) / (Omega_00 sqrt(4 pi)), where Omega_L
        # is the multipole-L part of the mask, Y_l = C^Omega_l / Omega_00^2 and Z_L = (2L + 1) C^Omega_L / (4 pi).
        ell_of_lm = hp.Alm.getlm(ell[-1])[0]
        X = np.array([hp.alm2cl(mask_lm, hp.map2alm(mask * hp.alm2map(mask_lm * (ell_of_lm == L), nside), iter=3))
                      for L in ell]).T
        M = M - 2 * X / (mask_00 * np.sqrt(4 * np.pi)) + np.outer(mask_cl / mask_00**2, (2 * ell + 1) * mask_cl / (4 * np.pi))
    pixwin2 = hp.pixwin(nside)**2
    in_band = (ell >= ell_edges[:-1, None]) & (ell < ell_edges[1:, None])
    binning = in_band / in_band.sum(1)[:, None] / pixwin2
    return dict(theory=binning @ M * pixwin2 / np.mean(mask**2), binning=binning, coupling=M, coupling2=coupling(mask**2),
                noise_shape=1 - ric * mask_cl / mask_00**2, pixwin2=pixwin2, mask=mask, ric=ric)


def measure(ra, dec, z, win, zbins, nomega, width=1, pairs=False, nz=None, w=None, dz=0.005):
    """Measure Z_ell(omega), or all Z_ell(omega, +-omega'), from a catalog.

    Parameters
    ----------
    ra, dec : (ngal,) arrays
        Angular positions in degrees. Galaxies in pixels where the mask vanishes are ignored.
    z : (ngal,) array
        Redshifts.
    win : dict
        Mask operators from window().
    zbins : (nbin + 1,) array
        Redshift bin edges.
    nomega : int
        Number of radial frequencies per bin, omega = m omega_f for m = 0, ..., nomega - 1, with omega_f = pi / (z1 - z0).
    width : int, optional
        Average over bands of `width` consecutive frequencies (band 0 is omega = 0 alone; see bands).
    pairs : bool, optional
        Return the cross-spectra of all pairs of frequency bands, Z(A, +B) and Z(A, -B) (see bin_omega and rows), instead of
        Z_ell(omega) only.
    nz : callable or None, optional
        Background density nbar(z) per unit redshift and steradian, with nbar_g(n, z) = nbar(z) Omega(n). If None
        (default), it is estimated from the catalog (requires window(..., ric=True)); otherwise use window(..., ric=False).
    w : callable or None, optional
        Radial weight w(z) of the galaxies (default 1).
    dz : float, optional
        Shell width used to estimate the normalization N from the catalog (nz=None).

    Returns
    -------
    array
        pairs=False: (nbin, n_omega_bands, nband), Z_ell(omega) per redshift bin, frequency band and multipole band.
        pairs=True: (nbin, nrow, nband), real and imaginary parts of the band pairs, with rows labelled by rows(n_omega_bands).
    """
    if (nz is None) != win['ric']:
        raise ValueError('estimate the background from the catalog (nz=None) with window(ric=True), or pass nz with ric=False')
    mask, weight = win['mask'], (lambda x: np.ones_like(x)) if w is None else w
    nside = hp.npix2nside(len(mask))
    lmax, pixarea, pix = 3 * nside - 1, hp.nside2pixarea(nside), hp.ang2pix(nside, ra, dec, lonlat=True)
    row, col = np.indices((nomega, nomega))
    Z = np.zeros((2, len(zbins) - 1, nomega, nomega, len(win['binning'])), complex)  # Z(w_i, w_j), Z(w_i, -w_j)
    for ibin, (z0, dz_bin) in enumerate(zip(zbins[:-1], np.diff(zbins))):
        inbin = (z >= z0) & (z < z0 + dz_bin) & (mask[pix] > 0)
        if not inbin.any():
            raise ValueError(f'no galaxies in redshift bin {ibin}')
        t, wg = z[inbin] - z0, weight(z[inbin])                         # redshift from the bin start, weights
        omega_f = np.pi / dz_bin
        start = np.exp(-1j * omega_f * z0 * np.arange(2 * nomega - 1))   # exp(-i omega z0), omega = m omega_f

        # frequency maps: sum_g w exp(-i omega z_g) per steradian, on the pixels that contain galaxies
        occupied, ipix = np.unique(pix[inbin], return_inverse=True)
        maps = np.zeros((nomega, len(mask)), complex)
        maps[:, occupied] = (nufft(ipix, t, nomega, dz_bin, wg) * start[:nomega]).T / pixarea

        # background [nbar w](z) Omega(n) and normalization N = int dz int dn/4pi nbar_g^2 w^2
        if nz is None:                                                    # nbar(z) from the catalog: N from shell sums
            nshell = max(1, round(dz_bin / dz))
            s1, s2 = (np.histogram(t, nshell, (0, dz_bin), weights=x)[0] for x in (wg, wg**2))
            N = np.mean(mask**2) * np.sum(s1**2 - s2) * nshell / dz_bin / (mask.sum() * pixarea)**2
            maps -= np.outer(maps.sum(1) / mask.sum(), mask)
        else:
            zz = np.linspace(z0, z0 + dz_bin, 10001)
            nw = nz(zz) * weight(zz)
            N = np.mean(mask**2) * np.trapezoid(nw**2, zz)
            maps -= np.outer(np.trapezoid(nw * np.exp(-1j * omega_f * np.outer(np.arange(nomega), zz)), zz), mask)

        # a_lm(omega) from the harmonic coefficients R, I of the real and imaginary parts of each map (healpy, m >= 0):
        # a_lm = P_lm for m >= 0 and (-1)^m Q*_l|m| for m < 0, with P = R + iI, Q = R - iI; for a_lm(-omega), P and Q swap.
        alm = hp.map2alm(np.concatenate([maps.real, maps.imag]), iter=3, pol=False) / np.sqrt(N)
        P, Q = alm[:nomega] + 1j * alm[nomega:], alm[:nomega] - 1j * alm[nomega:]
        cl = np.zeros((2, nomega, nomega, lmax + 1), complex)              # coupled spectra Z(w_i, w_j), Z(w_i, -w_j)
        for l in range(lmax + 1):
            m_idx = hp.Alm.getidx(lmax, l, np.arange(l + 1))
            p, q = P[:, m_idx], Q[:, m_idx]
            cl[0, ..., l] = (p @ p.conj().T + (q[:, 1:] @ q[:, 1:].conj().T).conj()) / (2 * l + 1)
            cl[1, ..., l] = (p @ q.conj().T + (q[:, 1:] @ p[:, 1:].conj().T).conj()) / (2 * l + 1)

        # Poisson noise sum_g w^2 exp(-i (omega_i -+ omega_j) z_g) / (4 pi N), a function of i -+ j only, times its shape in ell
        shot = nufft(np.zeros_like(ipix), t, 2 * nomega - 1, dz_bin, wg**2)[0] * start / (4 * np.pi * N)
        noise = np.array([np.where(row >= col, shot[abs(row - col)], shot[abs(row - col)].conj()), shot[row + col]])
        Z[:, ibin] = (cl - noise[..., None] * win['noise_shape']) @ win['binning'].T
    return bin_omega(Z[0], Z[1], width, pairs)


def bands(nomega, width):
    """Averaging matrix of the frequency bands.

    Parameters
    ----------
    nomega : int
        Number of frequencies.
    width : int
        Frequencies per band. Band 0 is {omega_0 = 0}; band n > 0 contains omega_m for m = (n - 1) width + 1, ..., n width
        (the last band may be narrower). There are n_omega_bands = 1 + ceil((nomega - 1) / width) bands.

    Returns
    -------
    (n_omega_bands, nomega) array
        Row n averages the frequencies of band n (so bands(nomega, width) @ omega gives the band centers).
    """
    band = np.r_[0, np.arange(nomega - 1) // width + 1]
    return (band == np.arange(band[-1] + 1)[:, None]) / np.bincount(band)[:, None]


def rows(n_omega_bands, pairs=True):
    """Labels of the rows of the band-pair data vector (measure with pairs=True).

    The independent pairs are Z(A, +B) for A <= B and Z(A, -B) for 0 < A <= B (Z(0, -B) = Z(0, B)^* since the omega = 0 map is
    real), each with real and imaginary parts, except Z(A, +A), which is real.

    Parameters
    ----------
    n_omega_bands : int
        Number of frequency bands.
    pairs : bool, optional
        If False, only the rows Z(A, +A).

    Returns
    -------
    (nrow, 4) int array
        (kind, A, B, part): kind 0 for Z(A, +B), 1 for Z(A, -B); part 0 for the real, 1 for the imaginary part.
    """
    return np.array([(kind, A, B, part) for kind in (0, 1) for A in range(kind, n_omega_bands) for B in range(A, n_omega_bands)
                     for part in (0, 1) if not (kind == 0 and A == B and part) and (pairs or kind == 0 and A == B)])


def bin_omega(Zpp, Zpm, width=1, pairs=False):
    """Frequency-binned data vector from the spectra of all frequency pairs.

    pairs=False gives Z_ell(omega) = Z(omega, omega) averaged over each frequency band. pairs=True gives the cross-spectra of the
    band-averaged maps, Z(A, +-B) = mean over omega in A and omega' in B of Z(omega, +-omega'); for width > 1 its Z(A, A) also
    contains the off-diagonal pairs within band A, so it differs from the band average of Z_ell(omega).

    Parameters
    ----------
    Zpp, Zpm : (nbin, nomega, nomega, nband) complex arrays
        Z(omega_i, omega_j) and Z(omega_i, -omega_j).
    width : int, optional
        Frequencies per band (see bands).
    pairs : bool, optional
        Return all band pairs instead of Z_ell(omega).

    Returns
    -------
    array
        pairs=False: (nbin, n_omega_bands, nband). pairs=True: (nbin, nrow, nband), with rows labelled by rows(n_omega_bands).
    """
    W = bands(Zpp.shape[1], width)
    if not pairs:
        return W @ np.einsum('bjjl->bjl', Zpp).real
    r = rows(len(W))
    Z = np.einsum('Aa,kzabl,Bb->kzABl', W, np.stack([Zpp, Zpm]), W)[r[:, 0], :, r[:, 1], r[:, 2]]
    return np.where(r[:, 3, None, None], Z.imag, Z.real).swapaxes(0, 1)
