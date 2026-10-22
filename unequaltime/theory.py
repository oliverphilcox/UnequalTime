"""Linear theory and Gaussian covariance of the unequal time power spectrum Z_ell(omega, omega')."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from .estimator import as_function, omega_bins, rows


def trapz_weights(x):
    """Trapezoid-rule weights on a uniform grid x."""
    return np.r_[0.5, np.ones(len(x) - 2), 0.5] * (x[1] - x[0])


def d2jl(l, x):
    """Spherical Bessel function j_l(x) and its second derivative j_l''(x), from the recurrence relations (finite at x = 0)."""
    jl = [spherical_jn(max(n, 0), x) for n in (l - 2, l, l + 2)]
    return jl[1], (l * (l - 1) * jl[0] / (2 * l - 1) - (l**2 / (2 * l - 1) + (l + 1)**2 / (2 * l + 3)) * jl[1]
                   + (l + 1) * (l + 2) * jl[2] / (2 * l + 3)) / (2 * l + 1)


def cosmology(cl, zmax, kmax):
    """Linear-theory inputs for spectra() from a computed CLASS instance, in Mpc/h units.

    Parameters
    ----------
    cl : classy.Class
        Computed CLASS instance, with output 'mPk' and P_k_max_1/Mpc >= h kmax.
    zmax : float
        Maximum redshift.
    kmax : float
        Maximum wavenumber [h/Mpc].

    Returns
    -------
    dict
        z, chi [Mpc/h] (comoving distance), D (growth factor, D(0) = 1), f (growth rate) on 3001 redshifts in [0, zmax], and
        k [h/Mpc], pk [(Mpc/h)^3] (linear matter power spectrum at z = 0) on 1000 wavenumbers in [1e-4, kmax].
    """
    h, z, k = cl.h(), np.linspace(0, zmax, 3001), np.logspace(-4, np.log10(kmax), 1000)
    return dict(z=z, chi=h * np.array([cl.comoving_distance(x) for x in z]),
                D=np.array([cl.scale_independent_growth_factor(x) for x in z]),
                f=np.array([cl.scale_independent_growth_factor_f(x) for x in z]),
                k=k, pk=h**3 * np.array([cl.pk_lin(h * x, 0) for x in k]))


def spectra(cosmo, b, nbar, zbins, nomega, lmax, w=None, ells=None, kmax=0.2, nk=1500, dchi=0.2, cross_bins=True, verbose=False):
    """Unmasked linear theory Z_ell(omega, omega') and Z_ell(omega, -omega') of all frequency pairs, in all redshift bins.

    Non-Limber, in the Kaiser approximation:

        Z_l(omega, omega') = (2/pi) int k^2 dk Delta_l(k, omega) Delta*_l(k, omega') P_L(k),
        Delta_l(k, omega) = N^{-1/2} int dz exp(-i omega z) nbar(z) w(z) [b(z) j_l(k chi(z)) - f(z) j_l''(k chi(z))] D(z),

    with N = int dz nbar^2 w^2 over the redshift bin (the unmasked normalization).

    Parameters
    ----------
    cosmo : dict
        Arrays 'z', 'chi' (comoving distance, Mpc/h), 'D' (linear growth factor, D(0) = 1), 'f' (growth rate) on a common
        redshift grid, and 'k' (h/Mpc), 'pk' (linear matter power spectrum at z = 0, (Mpc/h)^3), covering [1e-4, kmax], as
        returned by cosmology().
    b : callable
        Linear bias b(z).
    nbar : callable
        Background density nbar(z) per unit redshift and steradian (only its shape matters here).
    zbins : (nbin + 1,) array
        Redshift-bin edges.
    nomega : int
        Number of frequencies in each redshift bin.
    lmax : int
        Maximum multipole; use 3 nside - 1 of the mask for the expectation and covariance.
    w : callable or None, optional
        Radial weight w(z) (default 1), as in estimator.measure().
    ells : sorted int array, optional
        Multipoles at which the theory is computed, linearly interpolated in between and zero outside [min(ells), max(ells)]
        (so include 0 and lmax). Default: every multipole below 50, then every fourth, and lmax; np.arange(lmax + 1) computes all.
    kmax : float, optional
        Maximum wavenumber [h/Mpc].
    nk : int, optional
        Number of wavenumbers in (0, kmax].
    dchi : float, optional
        Radial integration step [Mpc/h].
    cross_bins : bool, optional
        Whether to compute the spectra between different redshift bins (needed for the covariance, not for the expectation).
    verbose : bool, optional
        Print progress.

    Returns
    -------
    Z_plus, Z_minus : (nbin, nomega, nbin, nomega, lmax + 1) complex arrays
        Z_plus[p, i, q, j, l] = <a_lm(omega_i) a*_lm(omega_j)> and Z_minus[p, i, q, j, l] = <a_lm(omega_i) a*_lm(-omega_j)>,
        with a_lm(omega_i) in redshift bin p and a_lm(omega_j) in redshift bin q (i.e. Z_l(omega_i, +-omega_j) for p = q); zero
        for p != q if cross_bins is False.
    """
    zbins = np.asarray(zbins, dtype=float)
    k = np.linspace(1e-4, kmax, nk)
    pk = np.exp(np.interp(np.log(k), np.log(cosmo['k']), np.log(cosmo['pk'])))
    k_weight = np.sqrt(2 / np.pi * k**2 * pk * trapz_weights(k))[:, None]                 # Z = sum_k Delta Delta^* k_weight^2
    weight = as_function(1. if w is None else w)

    def at(z, key):
        return np.interp(z, cosmo['z'], cosmo[key])
    kernels = []                                                                         # per redshift bin
    for z0, z1 in zip(zbins[:-1], zbins[1:]):
        z = np.linspace(z0, z1, int((at(z1, 'chi') - at(z0, 'chi')) / dchi) + 2)
        nw = nbar(z) * weight(z)
        # radial kernels dz exp(-i omega z) nbar w / N^{1/2}, as in the estimator
        dz = trapz_weights(z)
        radial = np.exp(-1j * np.pi * np.outer(np.arange(nomega), z) / (z1 - z0)) * dz * nw / np.sqrt(dz @ nw**2)
        kernels.append((np.outer(k, at(z, 'chi')), b(z) * at(z, 'D'), at(z, 'f') * at(z, 'D'), radial))
    ells = np.unique(np.r_[0:min(50, lmax + 1), 50:lmax:4, lmax]) if ells is None else np.asarray(ells)
    nbin = len(zbins) - 1
    if verbose:
        print(f'Computing the linear theory at {len(ells)} multipoles (k < {kmax} h/Mpc)', flush=True)
    x = np.arange(0, kmax * at(zbins[-1], 'chi') + 2e-3, 1e-3)                             # tabulation grid of j_l(x), j_l''(x)
    Z = np.zeros((2, nbin, nomega, nbin, nomega, len(ells)), complex)
    for il, l in enumerate(ells):
        jl, jl2 = d2jl(l, x)
        # Delta_l(k, omega) = int dz (radial kernel) [b j_l(k chi) - f j_l''(k chi)] D   (density + Kaiser term)
        delta = k_weight * np.concatenate([(bD * np.interp(kchi, x, jl) - fD * np.interp(kchi, x, jl2)) @ radial.T
                                           for kchi, bD, fD, radial in kernels], 1)
        for p, q in [(p, q) for p in range(nbin) for q in range(nbin) if cross_bins or p == q]:
            dp, dq = delta[:, p * nomega:(p + 1) * nomega], delta[:, q * nomega:(q + 1) * nomega]
            Z[:, p, :, q, :, il] = dp.T @ dq.conj(), dp.T @ dq
    Z = interp1d(ells, Z, axis=-1, bounds_error=False, fill_value=0)(np.arange(lmax + 1))
    return Z[0], Z[1]


def covariance(Z_plus, Z_minus, nbar, zbins, win, width=1, pairs=False, w=None, verbose=False):
    """Gaussian covariance of the data vector (see Survey.vector), including the correlations between redshift bins.

    Computed in the improved narrow-kernel approximation (Garcia-Garcia et al. 2019, arXiv:1906.11765), with the Poisson noise
    of the given background density and the mask (and radial integral constraint) of `win`. The approximation is not guaranteed
    to be positive definite; check its eigenvalues before inverting very large matrices.

    Parameters
    ----------
    Z_plus, Z_minus : (nbin, nomega, nbin, nomega, 3 nside) complex arrays
        Unmasked theory from spectra(), with cross_bins=True.
    nbar : callable
        Background density nbar(z) per unit redshift and steradian, with nbar_g = nbar(z) Omega(n), e.g. from
        estimator.background() (sets the shot noise).
    zbins : (nbin + 1,) array
        Redshift-bin edges.
    win : dict
        Mask operators from estimator.window().
    width, pairs, w : optional
        Number of frequencies in each omega-bin, whether to use all pairs of omega-bins, and the radial weight w(z), as in Survey.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    (ndata, ndata) array
        Covariance of the data vector: Z_ell(omega) (pairs=False) or the real and imaginary parts of the independent pairs of
        omega-bins (pairs=True, rows labelled by estimator.rows()), ordered as (redshift bin, omega-bin or row, ell-bin).
    """
    zbins = np.asarray(zbins, dtype=float)
    nbin, nomega, nl = Z_plus.shape[0], Z_plus.shape[1], Z_plus.shape[-1]
    mask_mean, mask2_mean, nell = np.mean(win['mask']), np.mean(win['mask']**2), len(win['binning'])
    bins = omega_bins(nomega, width if pairs else 1)             # pairs: bin-averaged before the Wick sums; Z(omega): after
    nB = len(bins)
    average = np.kron(np.eye(nbin), bins)                        # omega-bin averaging over (redshift bin, omega)

    # Poisson noise Z^shot(omega, +-omega') in the normalization of Z: <Omega> int dz exp(-i (omega -+ omega') z) nbar w^2 / N
    weight = as_function(1. if w is None else w)
    noise = np.zeros((2, nbin * nomega, nbin * nomega), complex)
    for ibin, (z0, z1) in enumerate(zip(zbins[:-1], zbins[1:])):
        z = np.linspace(z0, z1, 10001)
        n, wz = nbar(z), weight(z)
        phase = np.exp(-1j * np.pi * np.outer(np.arange(nomega), z) / (z1 - z0))
        integrand = trapz_weights(z) * n * wz**2 * mask_mean / (mask2_mean * np.trapezoid((n * wz)**2, z))
        sl = slice(ibin * nomega, (ibin + 1) * nomega)
        noise[:, sl, sl] = (phase * integrand) @ phase.conj().T, (phase * integrand) @ phase.T

    # Spectra S of the signed omega-bins a = (+-, redshift bin, A), with a_lm(-omega) = (-1)^m a*_l-m(omega), so that
    # <a a'^*> = [[Z_plus, Z_minus], [Z_minus^*, Z_plus^*]], of shape (ell, a, a'): coupled through the mask (signal) plus the
    # noise times its shape, and divided by <Omega^2> (iNKA).
    def signed(zp, zm):
        return np.block([[zp, zm], [zm.conj(), zp.conj()]])

    def bin_average(x):
        return average @ x @ average.T
    S = signed(*(bin_average(x.reshape(nbin * nomega, nbin * nomega, nl).transpose(2, 0, 1)) for x in (Z_plus, Z_minus)))
    S = (np.tensordot(win['coupling'] * win['pixwin2'], S, 1) / mask2_mean
         + win['noise_shape'][:, None, None] * signed(*map(bin_average, noise))) / mask2_mean
    used = win['binning'].any(0)                                 # multipoles entering the ell-bins
    # DS[a, b, i, l] = binning[i, l] S_l(a, b); DSX = DS @ M2_ll' / (2l' + 1), the iNKA four-point coupling
    DS = win['binning'][:, used] * S[used].transpose(1, 2, 0)[:, :, None]
    DSX = DS @ (win['coupling2'] / (2 * np.arange(nl) + 1))[np.ix_(used, used)]

    # Each row is Re(c Z(a, b)) with c = 1 (real part) or -i (imaginary part), a = (+, bin, A), b = (+-, bin, B). By Wick,
    #   <Z(a, b) Z(c, d)^*> = S_ac S_bd^* + S_a,-d S_b,-c^*,   <Z(a, b) Z(c, d)> = S_ad S_bc^* + S_a,-c S_b,-d^*,
    #   Cov(Re c X, Re c' Y) = Re(c c'^* <X Y^*> + c c' <X Y>) / 2.
    labels = rows(nB, pairs)
    if verbose:
        print(f'Computing the covariance of {nbin * len(labels) * nell} elements before omega-binning', flush=True)
    first = np.concatenate([ibin * nB + labels[:, 1] for ibin in range(nbin)])
    second = np.concatenate([(labels[:, 0] * nbin + ibin) * nB + labels[:, 2] for ibin in range(nbin)])
    part = np.tile(np.where(labels[:, 3], -1j, 1), nbin)[:, None, None]

    def flip(a):                                                 # index of the opposite-sign omega-bin
        return (a + nbin * nB) % (2 * nbin * nB)

    def herm(x):
        return x.conj().swapaxes(-1, -2)
    unique_pairs, pair_of_row = np.unique(np.c_[first, second], axis=0, return_inverse=True)
    cov = np.zeros((len(first), len(first), nell, nell))
    for ip, (a, b) in enumerate(unique_pairs):
        XYc = DSX[a, first] @ herm(DS[b, second]) + DSX[a, flip(second)] @ herm(DS[b, flip(first)])
        XY = DSX[a, second] @ herm(DS[b, first]) + DSX[a, flip(first)] @ herm(DS[b, flip(second)])
        for r in np.flatnonzero(pair_of_row == ip):
            cov[r] = 0.5 * np.real(part[r] * part.conj() * XYc + part[r] * part * XY)
    cov = cov.transpose(0, 2, 1, 3).reshape(len(first) * nell, -1)
    if not pairs:                                                # average Z(omega) over the omega-bins
        project = np.kron(np.kron(np.eye(nbin), omega_bins(nomega, width)), np.eye(nell))
        cov = project @ cov @ project.T
    return 0.5 * (cov + cov.T)
