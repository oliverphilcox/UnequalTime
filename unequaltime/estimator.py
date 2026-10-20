"""Estimator of the unequal-time angular power spectra Z_ell(omega, omega') of a (RA, Dec, z) catalogue.

In each redshift bin [z0, z0 + L) the galaxy density minus its mean is Fourier transformed along redshift,

    F_omega(n) = [sum_{g in pixel n} exp(-i omega (z_g - z0)) / pixel area - background] / sqrt(N),   omega_m = pi m / L,

and Z_ell(omega, omega') = sum_m F_lm(omega) F_lm(omega')^* / (2 ell + 1) - shot noise, binned in ell. The spectra are coupled
pseudo-C_ell: the mask is not deconvolved but applied to the theory, using mode-coupling matrices from NaMaster.
"""
import numpy as np
import healpy as hp
import pymaster as nmt


def nufft(ipix, t, nomega, L, support=7):
    """Sum of exp(-i omega_m t) over the points in each pixel, by a non-uniform FFT.

    The points are spread onto a periodic grid of 4 nomega cells on [0, 2L) with an exponential-of-semicircle kernel, Fourier
    transformed, and divided by the kernel transform (Barnett et al. 2019, arXiv:1808.06736); relative accuracy ~1e-7.

    Parameters
    ----------
    ipix : (npoint,) int array
        Pixel index of each point, in 0, 1, ..., npix - 1.
    t : (npoint,) array
        Positions, in [0, L).
    nomega : int
        Number of frequencies omega_m = pi m / L, m = 0, ..., nomega - 1.
    L : float
        Length of the interval.
    support : int, optional
        Width of the spreading kernel in grid cells.

    Returns
    -------
    (npix, nomega) complex array
        sum over the points in pixel p of exp(-i omega_m t).
    """
    kernel = lambda x: np.exp(2.3 * support * (np.sqrt(np.maximum(1 - x**2, 0)) - 1))      # x in [-1, 1]
    ngrid = 4 * nomega
    u = t * ngrid / (2 * L)                                                                # positions in grid cells
    cells = np.floor(u - support / 2).astype(int)[:, None] + 1 + np.arange(support)        # nearest `support` cells
    weights = kernel((u[:, None] - cells) / (support / 2))
    grid = np.bincount((ipix[:, None] * ngrid + cells % ngrid).ravel(), weights.ravel(), (ipix.max() + 1) * ngrid)
    y = np.linspace(-1, 1, 2001)
    kernel_ft = support / 2 * np.trapezoid(kernel(y) * np.cos(np.pi * support / ngrid * np.outer(np.arange(nomega), y)), y)
    return np.fft.rfft(grid.reshape(-1, ngrid))[:, :nomega] / kernel_ft


def window(mask, ell_edges, ric=True):
    """Mask operators for the estimator, its theory and its covariance.

    Parameters
    ----------
    mask : (npix,) array
        HEALPix map (RING ordering) of the angular selection Omega >= 0, in any normalisation. Its resolution nside sets the
        multipole range, ell = 0, ..., 3 nside - 1.
    ell_edges : (nband + 1,) int array
        Bandpower edges: band i contains ell_edges[i] <= ell < ell_edges[i + 1].
    ric : bool, optional
        Whether the background n(z) will be estimated from the catalogue (measure with nz=None). This imposes a radial integral
        constraint (the masked mean of each map vanishes), which is included in the coupling matrix and the shot noise.

    Returns
    -------
    dict
        A : (nband, 3 nside) array, theory operator taking full-sky spectra to the expected bandpowers (see theory.mean);
        D : (nband, 3 nside) array, binning of coupled spectra with the pixel window removed;
        Mt : (3 nside, 3 nside) array, mode-coupling matrix of the mask (with the integral constraint if ric);
        M2 : (3 nside, 3 nside) array, mode-coupling matrix of mask^2 (for the covariance);
        s : (3 nside,) array, multipole dependence of the shot noise (1 without the integral constraint);
        b2 : (3 nside,) array, pixel window squared;
        mask, ric : the inputs.
    """
    nside, ell_edges = hp.npix2nside(len(mask)), np.asarray(ell_edges)
    ell = np.arange(3 * nside)
    assert ell_edges[-1] <= len(ell), 'bandpowers must lie below 3 nside'

    def coupling(w):                                                     # NaMaster mode-coupling matrix of the window w
        f = nmt.NmtField(w, None, spin=0)
        return nmt.NmtWorkspace.from_fields(f, f, nmt.NmtBin.from_lmax_linear(ell[-1], 1)).get_coupling_matrix()
    Mt, mask_lm = coupling(mask), hp.map2alm(mask, iter=3)
    mask_cl, mask_00 = hp.alm2cl(mask_lm), mask_lm[0].real
    if ric:
        # The data-estimated background subtracts Omega int F / int Omega from each map (a_lm -> a_lm - Omega_lm a_00 / Omega_00):
        # Mt = M - 2 X + Y Z^T, X[l, L] = C_l(Omega, Omega Omega_L) / (Omega_00 sqrt(4 pi)) with Omega_L the multipole-L part of
        # the mask, Y_l = C^Omega_l / Omega_00^2, Z_L = (2L + 1) C^Omega_L / (4 pi).
        ell_of_lm = hp.Alm.getlm(ell[-1])[0]
        mask_L = lambda L: hp.alm2map(mask_lm * (ell_of_lm == L), nside)
        X = np.array([hp.alm2cl(mask_lm, hp.map2alm(mask * mask_L(L), iter=3)) for L in ell]).T
        Mt = Mt - 2 * X / (mask_00 * np.sqrt(4 * np.pi)) + np.outer(mask_cl / mask_00**2, (2 * ell + 1) * mask_cl / (4 * np.pi))
    b2 = hp.pixwin(nside)**2
    in_band = (ell >= ell_edges[:-1, None]) & (ell < ell_edges[1:, None])
    D = in_band / in_band.sum(1)[:, None] / b2
    return dict(A=D @ Mt * b2 / np.mean(mask**2), D=D, Mt=Mt, M2=coupling(mask**2), s=1 - ric * mask_cl / mask_00**2, b2=b2,
                mask=mask, ric=ric)


def measure(ra, dec, z, win, zbins, nomega, width=1, pairs=False, nz=None, dz=0.005):
    """Measure Z_ell(omega), or all Z_ell(omega, +-omega'), from a catalogue.

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
        Number of radial frequencies per bin, omega_m = pi m / (z1 - z0) for m = 0, ..., nomega - 1, with phases
        exp(-i omega (z - z0)) measured from the bin start z0.
    width : int, optional
        Average over bands of `width` consecutive frequencies (band 0 is omega = 0 alone; see bands).
    pairs : bool, optional
        Return all band pairs Z(A, +B) and Z(A, -B) (see rows) instead of Z_ell(omega) only.
    nz : callable or None, optional
        Background density: nz(z) is the mean number of galaxies per unit redshift and steradian, n_g = nz(z) Omega(n). If None
        (default), it is estimated from the catalogue (requires window(..., ric=True)); otherwise use window(..., ric=False).
    dz : float, optional
        Shell width used to estimate the normalisation N from the catalogue (nz=None).

    Returns
    -------
    array
        pairs=False: (nbin, n_omega_bands, nband), Z_ell(omega) per redshift bin, omega band and ell band.
        pairs=True: (nbin, nrow, nband), real and imaginary parts of the band pairs, with rows given by rows(n_omega_bands).
    """
    assert (nz is None) == win['ric'], 'the background is estimated from the catalogue (nz=None) iff window(ric=True)'
    mask = win['mask']
    nside = hp.npix2nside(len(mask))
    lmax, pixarea, pix = 3 * nside - 1, hp.nside2pixarea(nside), hp.ang2pix(nside, ra, dec, lonlat=True)
    i, j = np.indices((nomega, nomega))
    Z = np.zeros((2, len(zbins) - 1, nomega, nomega, len(win['D'])), complex)     # Z(w_i, w_j), Z(w_i, -w_j)
    for b, (z0, L) in enumerate(zip(zbins[:-1], np.diff(zbins))):
        inbin = (z >= z0) & (z < z0 + L) & (mask[pix] > 0)
        t = z[inbin] - z0

        # frequency maps F_omega: sum_g exp(-i omega t_g) per steradian, on the pixels that contain galaxies
        occupied, ipix = np.unique(pix[inbin], return_inverse=True)
        F = np.zeros((nomega, len(mask)), complex)
        F[:, occupied] = nufft(ipix, t, nomega, L).T / pixarea

        # background nbar(z) Omega(n) and normalisation N = int dz int dn/4pi nbar_g^2
        if nz is None:                                                  # nbar(z) from the catalogue: N from shell counts
            counts = np.histogram(t, max(1, round(L / dz)), (0, L))[0]
            N = np.mean(mask**2) * np.sum(counts**2 - counts) * len(counts) / L / (mask.sum() * pixarea)**2
            F -= np.outer(F.sum(1) / mask.sum(), mask)
        else:
            zz = np.linspace(z0, z0 + L, 10001)
            n = nz(zz)
            N = np.mean(mask**2) * np.trapezoid(n**2, zz)
            F -= np.outer(np.trapezoid(n * np.exp(-1j * np.pi * np.outer(np.arange(nomega), zz - z0) / L), zz), mask)

        # Harmonic coefficients of the complex maps from those of Re F and Im F (healpy, m >= 0): F_lm = P_lm for m >= 0 and
        # (-1)^m Q*_l|m| for m < 0, with P = R + iI, Q = R - iI; similarly for F^* with P and Q swapped.
        alm = hp.map2alm(np.concatenate([F.real, F.imag]), iter=3, pol=False) / np.sqrt(N)
        P, Q = alm[:nomega] + 1j * alm[nomega:], alm[:nomega] - 1j * alm[nomega:]
        cl = np.zeros((2, nomega, nomega, lmax + 1), complex)              # coupled spectra of (F_i, F_j) and (F_i, F_j^*)
        for l in range(lmax + 1):
            m_idx = hp.Alm.getidx(lmax, l, np.arange(l + 1))
            p, q = P[:, m_idx], Q[:, m_idx]
            cl[0, ..., l] = (p @ p.conj().T + (q[:, 1:] @ q[:, 1:].conj().T).conj()) / (2 * l + 1)
            cl[1, ..., l] = (p @ q.conj().T + (q[:, 1:] @ p[:, 1:].conj().T).conj()) / (2 * l + 1)

        # Poisson noise sum_g exp(-i (omega_i -+ omega_j) t_g) / (4 pi N) (a function of i -+ j), times its shape s_l
        shot = nufft(np.zeros_like(ipix), t, 2 * nomega - 1, L)[0] / (4 * np.pi * N)
        noise = np.array([np.where(i >= j, shot[abs(i - j)], shot[abs(i - j)].conj()), shot[i + j]])
        Z[:, b] = (cl - noise[..., None] * win['s']) @ win['D'].T
    return bin_omega(Z[0], Z[1], width, pairs)


def bands(nomega, width):
    """Averaging matrix of the frequency bands.

    Parameters
    ----------
    nomega : int
        Number of frequencies.
    width : int
        Frequencies per band. Band 0 is {omega_0 = 0}; band n > 0 contains omega_m for m = (n - 1) width + 1, ..., n width.

    Returns
    -------
    (n_omega_bands, nomega) array
        Row n averages the frequencies of band n.
    """
    band = np.r_[0, np.arange(nomega - 1) // width + 1]
    return (band == np.arange(band[-1] + 1)[:, None]) / np.bincount(band)[:, None]


def rows(n_omega_bands, pairs=True):
    """Labels of the rows of the band-pair data vector (measure with pairs=True).

    The independent pairs are Z(A, +B) for A <= B and Z(A, -B) for 0 < A <= B (Z(0, -B) = Z(0, B)^* since the omega = 0 map is
    real), each with real and imaginary parts, except for the real Z(A, +A).

    Parameters
    ----------
    n_omega_bands : int
        Number of frequency bands.
    pairs : bool, optional
        If False, only the rows Z(A, +A) (i.e. Z_ell(omega) for width 1).

    Returns
    -------
    (nrow, 4) int array
        (kind, A, B, part): kind 0 for Z(A, +B), 1 for Z(A, -B); part 0 for the real, 1 for the imaginary part.
    """
    return np.array([(kind, A, B, part) for kind in (0, 1) for A in range(kind, n_omega_bands) for B in range(A, n_omega_bands)
                     for part in (0, 1) if not (kind == 0 and A == B and part) and (pairs or kind == 0 and A == B)])


def bin_omega(Zpp, Zpm, width=1, pairs=False):
    """Data vector from all frequency pairs.

    Parameters
    ----------
    Zpp, Zpm : (nbin, nomega, nomega, nband) complex arrays
        Z(omega_i, omega_j) and Z(omega_i, -omega_j).
    width : int, optional
        Frequencies per band (see bands).
    pairs : bool, optional
        Return the band averages of all pairs (rows) instead of Z_ell(omega).

    Returns
    -------
    array
        pairs=False: (nbin, n_omega_bands, nband), band averages of Z_ell(omega) = Z(omega, omega).
        pairs=True: (nbin, nrow, nband), band averages of Z(A, +-B), with rows given by rows(n_omega_bands).
    """
    W = bands(Zpp.shape[1], width)
    if not pairs:
        return W @ np.einsum('bjjl->bjl', Zpp).real
    r = rows(len(W))
    Z = np.einsum('Aa,kzabl,Bb->kzABl', W, np.stack([Zpp, Zpm]), W)[r[:, 0], :, r[:, 1], r[:, 2]]
    return np.where(r[:, 3, None, None], Z.imag, Z.real).swapaxes(0, 1)
