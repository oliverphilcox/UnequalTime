"""Estimator of the unequal time power spectrum Z_ell(omega, omega') of a (RA, Dec, z) catalog.

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


def as_function(x):
    """A function of redshift, from a callable or a constant."""
    return x if callable(x) else (lambda z: np.full(np.shape(z), float(x)))


def nshell(nomega):
    """Number of redshift shells per redshift bin used to estimate nbar(z) from the data: shells of width Delta z / (nomega - 1),
    the radial resolution of the frequencies omega <= (nomega - 1) pi / Delta z."""
    return max(1, nomega - 1)


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


def window(mask, ell_edges, ric=True, verbose=False):
    """Mask operators for the estimator, its theory and its covariance.

    Parameters
    ----------
    mask : (npix,) array
        HEALPix map (RING ordering) of the angular mask Omega >= 0, in any normalization. Its resolution nside sets the multipole
        range, ell = 0, ..., 3 nside - 1.
    ell_edges : (nell + 1,) int array
        Edges of the ell-bins: bin i contains ell_edges[i] <= ell < ell_edges[i + 1], with ell_edges[-1] <= 3 nside.
    ric : bool, optional
        Whether nbar(z) will be estimated from the catalog (measure with nbar=None). This imposes a radial integral constraint
        (the masked monopole of each map vanishes), which is included in the mode coupling and the shot noise.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    dict
        theory : (nell, 3 nside) array, operator taking unmasked spectra to the expected ell-binned spectra;
        binning : (nell, 3 nside) array, ell-binning of coupled spectra with the pixel window removed;
        coupling : (3 nside, 3 nside) array, mode-coupling matrix of the mask (with the integral constraint if ric);
        coupling2 : (3 nside, 3 nside) array, mode-coupling matrix of mask^2 (for the covariance);
        noise_shape : (3 nside,) array, multipole dependence of the shot noise (1 without the integral constraint);
        pixwin2 : (3 nside,) array, pixel window squared;
        mask, ric : the inputs.
    """
    mask, ell_edges = np.asarray(mask, dtype=float), np.asarray(ell_edges)
    nside = hp.npix2nside(len(mask))
    ell = np.arange(3 * nside)
    if ell_edges[-1] > len(ell):
        raise ValueError(f'ell_edges must not exceed 3 nside = {len(ell)}')

    if verbose:
        print(f'Computing the mode-coupling matrices of the mask (nside {nside})', flush=True)

    def coupling(m):                                                     # NaMaster mode-coupling matrix of the mask m
        field = nmt.NmtField(m, None, spin=0)
        return nmt.NmtWorkspace.from_fields(field, field, nmt.NmtBin.from_lmax_linear(ell[-1], 1)).get_coupling_matrix()
    M, mask_lm = coupling(mask), hp.map2alm(mask, iter=3)
    mask_cl, mask_00 = hp.alm2cl(mask_lm), mask_lm[0].real
    if ric:
        if verbose:
            print(f'Computing the radial integral constraint ({len(ell)} harmonic transforms)', flush=True)
        # The data-estimated background nulls the masked monopole of each map, a_lm -> a_lm - (Omega_lm / Omega_00) a_00,
        # so the coupling becomes M - 2 X + Y Z^T, with X[l, L] = C_l(Omega, Omega Omega_L) / (Omega_00 sqrt(4 pi)), where Omega_L
        # is the multipole-L part of the mask, Y_l = C^Omega_l / Omega_00^2 and Z_L = (2L + 1) C^Omega_L / (4 pi).
        ell_of_lm = hp.Alm.getlm(ell[-1])[0]
        X = np.array([hp.alm2cl(mask_lm, hp.map2alm(mask * hp.alm2map(mask_lm * (ell_of_lm == L), nside), iter=3))
                      for L in ell]).T
        M = M - 2 * X / (mask_00 * np.sqrt(4 * np.pi)) + np.outer(mask_cl / mask_00**2, (2 * ell + 1) * mask_cl / (4 * np.pi))
    pixwin2 = hp.pixwin(nside)**2
    in_bin = (ell >= ell_edges[:-1, None]) & (ell < ell_edges[1:, None])
    binning = in_bin / in_bin.sum(1)[:, None] / pixwin2
    return dict(theory=binning @ M * pixwin2 / np.mean(mask**2), binning=binning, coupling=M, coupling2=coupling(mask**2),
                noise_shape=1 - ric * mask_cl / mask_00**2, pixwin2=pixwin2, mask=mask, ric=ric)


def background(ra, dec, z, win, zbins, nomega):
    """Background density nbar^obs(z) estimated from the catalog, as in measure() with nbar=None.

    nbar^obs(z) = int dn n_g(n, z) / int dn Omega(n), in the same redshift shells (nomega - 1 per redshift bin; see nshell) and
    for the same galaxies (those in pixels with Omega > 0) as used by measure(). Pass the result as `nbar` to theory.spectra()
    and theory.covariance(); in measure(), the data-estimated background is used with nbar=None and window(..., ric=True), which
    includes the radial integral constraint.

    Parameters
    ----------
    ra, dec : (ngal,) arrays
        Angular positions in degrees.
    z : (ngal,) array
        Redshifts.
    win : dict
        Mask operators from window().
    zbins : (nbin + 1,) array
        Redshift-bin edges.
    nomega : int
        Number of frequencies in each redshift bin, as in measure() (sets the shell width).

    Returns
    -------
    callable
        nbar(z) per unit redshift and steradian: piecewise constant in the shells, zero outside the redshift bins.
    """
    mask, zbins = win['mask'], np.asarray(zbins, dtype=float)
    nside = hp.npix2nside(len(mask))
    z = np.asarray(z)[mask[hp.ang2pix(nside, ra, dec, lonlat=True)] > 0]
    area = mask.sum() * hp.nside2pixarea(nside)                           # int dn Omega(n)
    edges, density = [], []
    for z0, dz_bin in zip(zbins[:-1], np.diff(zbins)):
        n = nshell(nomega)
        counts = np.histogram(z[(z >= z0) & (z < z0 + dz_bin)] - z0, n, (0, dz_bin))[0]
        edges.append(z0 + np.arange(n) * dz_bin / n)
        density.append(counts * n / dz_bin / area)
    edges, density = np.concatenate(edges + [zbins[-1:]]), np.concatenate(density)

    def nbar(x):                                                       # shells [edge_i, edge_i+1), the last including zbins[-1]
        shell = np.where(np.asarray(x) == edges[-1], len(density) - 1, np.searchsorted(edges, x, side='right') - 1)
        return np.where((shell >= 0) & (shell < len(density)), density[np.clip(shell, 0, len(density) - 1)], 0.)
    nbar.from_catalog = True
    return nbar


def measure(ra, dec, z, win, zbins, nomega, nbar=None, w=None, verbose=False):
    """Measure Z_ell(omega, omega') and Z_ell(omega, -omega') for all pairs of frequencies from a catalog.

    Parameters
    ----------
    ra, dec : (ngal,) arrays
        Angular positions in degrees. Galaxies in pixels where the mask vanishes are ignored.
    z : (ngal,) array
        Redshifts.
    win : dict
        Mask operators from window().
    zbins : (nbin + 1,) array
        Redshift-bin edges.
    nomega : int
        Number of frequencies in each redshift bin, omega = m omega_f for m = 0, ..., nomega - 1, with omega_f = pi / (z1 - z0).
    nbar : callable or None, optional
        If None (default), the background is estimated from the catalog, nbar^obs(z), including the radial integral constraint
        (requires window(..., ric=True); see background()). Otherwise, a known background density nbar(z) per unit redshift and
        steradian, with nbar_g(n, z) = nbar(z) Omega(n) (requires window(..., ric=False)).
    w : callable or None, optional
        Radial weight w(z) of the galaxies (default 1).
    verbose : bool, optional
        Print progress.

    Returns
    -------
    Z_plus, Z_minus : (nbin, nomega, nomega, nell) complex arrays
        Z_ell(omega_i, omega_j) and Z_ell(omega_i, -omega_j) in each redshift bin and ell-bin (see bin_omega).
    """
    if getattr(nbar, 'from_catalog', False):
        raise ValueError('a background estimated from the catalog is applied with nbar=None and window(ric=True), which '
                         'includes the radial integral constraint')
    if (nbar is None) != win['ric']:
        raise ValueError('estimate the background from the catalog (nbar=None) with window(ric=True), '
                         'or pass nbar with window(ric=False)')
    ra, dec, z, zbins = (np.asarray(x, dtype=float) for x in (ra, dec, z, zbins))
    mask, weight = win['mask'], as_function(1. if w is None else w)
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
        if verbose:
            print(f'Redshift bin {ibin}: assigning {len(t)} galaxies to the grid and transforming to {nomega} frequencies',
                  flush=True)
        occupied, ipix = np.unique(pix[inbin], return_inverse=True)
        maps = np.zeros((nomega, len(mask)), complex)
        maps[:, occupied] = (nufft(ipix, t, nomega, dz_bin, wg) * start[:nomega]).T / pixarea

        # background [nbar w](z) Omega(n) and normalization N = int dz int dn/4pi nbar_g^2 w^2
        if nbar is None:                                                  # nbar(z) from the catalog: N from shell sums
            n = nshell(nomega)
            s1, s2 = (np.histogram(t, n, (0, dz_bin), weights=x)[0] for x in (wg, wg**2))
            N = np.mean(mask**2) * np.sum(s1**2 - s2) * n / dz_bin / (mask.sum() * pixarea)**2
            maps -= np.outer(maps.sum(1) / mask.sum(), mask)
        else:
            zz = np.linspace(z0, z0 + dz_bin, 10001)
            nw = nbar(zz) * weight(zz)
            N = np.mean(mask**2) * np.trapezoid(nw**2, zz)
            maps -= np.outer(np.trapezoid(nw * np.exp(-1j * omega_f * np.outer(np.arange(nomega), zz)), zz), mask)

        # a_lm(omega) from the harmonic coefficients R, I of the real and imaginary parts of each map (healpy, m >= 0):
        # a_lm = P_lm for m >= 0 and (-1)^m Q*_l|m| for m < 0, with P = R + iI, Q = R - iI; for a_lm(-omega), P and Q swap.
        if verbose:
            print(f'Redshift bin {ibin}: computing {2 * nomega} spherical harmonic transforms and the power spectra', flush=True)
        alm = hp.map2alm(np.concatenate([maps.real, maps.imag]), iter=3, pol=False) / np.sqrt(N)
        P, Q = alm[:nomega] + 1j * alm[nomega:], alm[:nomega] - 1j * alm[nomega:]
        coupled = np.zeros((2, nomega, nomega, lmax + 1), complex)         # coupled spectra Z(w_i, w_j), Z(w_i, -w_j)
        for l in range(lmax + 1):
            m_idx = hp.Alm.getidx(lmax, l, np.arange(l + 1))
            p, q = P[:, m_idx], Q[:, m_idx]
            coupled[0, ..., l] = (p @ p.conj().T + (q[:, 1:] @ q[:, 1:].conj().T).conj()) / (2 * l + 1)
            coupled[1, ..., l] = (p @ q.conj().T + (q[:, 1:] @ p[:, 1:].conj().T).conj()) / (2 * l + 1)

        # Poisson noise sum_g w^2 exp(-i (omega_i -+ omega_j) z_g) / (4 pi N), a function of i -+ j only, times its shape in ell
        shot = nufft(np.zeros_like(ipix), t, 2 * nomega - 1, dz_bin, wg**2)[0] * start / (4 * np.pi * N)
        noise = np.array([np.where(row >= col, shot[abs(row - col)], shot[abs(row - col)].conj()), shot[row + col]])
        Z[:, ibin] = (coupled - noise[..., None] * win['noise_shape']) @ win['binning'].T
    return Z[0], Z[1]


def omega_bins(nomega, width):
    """Averaging matrix of the omega-bins.

    Parameters
    ----------
    nomega : int
        Number of frequencies.
    width : int
        Number of frequencies in each omega-bin. Bin 0 is {omega_0 = 0}; bin n > 0 contains omega_m for
        m = (n - 1) width + 1, ..., n width (the last bin may be narrower), so there are 1 + ceil((nomega - 1) / width) bins.

    Returns
    -------
    (n_omega_bins, nomega) array
        Row n averages the frequencies of omega-bin n (so omega_bins(nomega, width) @ omega gives the bin centers).
    """
    if width < 1:
        raise ValueError('width must be at least 1')
    which = np.r_[0, np.arange(nomega - 1) // width + 1]
    return (which == np.arange(which[-1] + 1)[:, None]) / np.bincount(which)[:, None]


def rows(n_omega_bins, pairs=True):
    """Labels of the rows of the data vector of omega-bin pairs (see pair_vector).

    The independent pairs are Z(A, +B) for A <= B and Z(A, -B) for 0 < A <= B (Z(0, -B) = Z(0, B)^* since the omega = 0 map is
    real), each with real and imaginary parts, except Z(A, +A), which is real.

    Parameters
    ----------
    n_omega_bins : int
        Number of omega-bins.
    pairs : bool, optional
        If False, only the rows Z(A, +A).

    Returns
    -------
    (nrow, 4) int array
        (kind, A, B, part): kind 0 for Z(A, +B), 1 for Z(A, -B); part 0 for the real, 1 for the imaginary part.
    """
    return np.array([(kind, A, B, part) for kind in (0, 1) for A in range(kind, n_omega_bins) for B in range(A, n_omega_bins)
                     for part in (0, 1) if not (kind == 0 and A == B and part) and (pairs or kind == 0 and A == B)])


def bin_omega(Z_plus, Z_minus, width=1, pairs=False):
    """Average the spectra of all frequency pairs over omega-bins.

    pairs=False gives Z_ell(omega) = Z_ell(omega, omega) averaged within each omega-bin. pairs=True gives the cross-spectra of
    the bin-averaged fields, Z_ell(omega_A, +-omega_B) = mean over omega in A and omega' in B of Z_ell(omega, +-omega'); for
    width > 1, Z_ell(omega_A, omega_A) also contains the off-diagonal pairs within bin A, so it differs from the bin average of
    Z_ell(omega).

    Parameters
    ----------
    Z_plus, Z_minus : (nbin, nomega, nomega, nell) complex arrays
        Z_ell(omega_i, omega_j) and Z_ell(omega_i, -omega_j).
    width : int, optional
        Number of frequencies in each omega-bin (see omega_bins).
    pairs : bool, optional
        Return all pairs of omega-bins instead of Z_ell(omega).

    Returns
    -------
    array or tuple of arrays
        pairs=False: (nbin, n_omega_bins, nell) real array, Z_ell(omega).
        pairs=True: two (nbin, n_omega_bins, n_omega_bins, nell) complex arrays, Z_ell(omega_A, omega_B) and
        Z_ell(omega_A, -omega_B).
    """
    W = omega_bins(Z_plus.shape[1], width)
    if not pairs:
        return W @ np.einsum('bjjl->bjl', Z_plus).real
    return tuple(np.einsum('Aa,zabl,Bb->zABl', W, x, W) for x in (Z_plus, Z_minus))


def pair_vector(Z_plus, Z_minus):
    """Real data vector of the independent pairs of omega-bins.

    Parameters
    ----------
    Z_plus, Z_minus : (nbin, n_omega_bins, n_omega_bins, nell) complex arrays
        Z_ell(omega_A, omega_B) and Z_ell(omega_A, -omega_B), from bin_omega(..., pairs=True).

    Returns
    -------
    (nbin, nrow, nell) array
        Real and imaginary parts of the independent pairs, with rows labelled by rows(n_omega_bins).
    """
    labels = rows(Z_plus.shape[1])
    Z = np.stack([Z_plus, Z_minus])[labels[:, 0], :, labels[:, 1], labels[:, 2]]
    return np.where(labels[:, 3, None, None], Z.imag, Z.real).swapaxes(0, 1)
