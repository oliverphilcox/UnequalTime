"""Linear theory and Gaussian covariance of the unequal-time angular power spectra Z_ell(omega, omega')."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from .estimator import bands, rows, bin_omega


def trapz_weights(x):
    """Trapezoid-rule weights on a uniform grid x."""
    return np.r_[0.5, np.ones(len(x) - 2), 0.5] * (x[1] - x[0])


def spectra(cosmo, b, nz, zbins, nomega, lmax, ells=None, kmax=0.2, nk=1500, dchi=0.2):
    """Full-sky linear theory of the frequency maps: G = <a_i a_j^*> and Gt = <a_i a_j> for i = (redshift bin, omega_m).

    Non-Limber, with the Kaiser redshift-space term:

        G_l(i, j) = (2/pi) int dk k^2 P(k) Delta_i(k) Delta_j(k)^*,
        Delta_i(k) = int dz w_i(z) [b D j_l(k chi) - f D j_l''(k chi)],

    and Gt with Delta_j instead of Delta_j^*, where w_i(z) = nz(z) exp(-i omega_m (z - z0)) / sqrt(int dz nz^2) are the radial
    weights of the estimator in redshift bin [z0, z1).

    Parameters
    ----------
    cosmo : dict
        Arrays 'z', 'chi' (comoving distance), 'D' (linear growth factor, D(0) = 1), 'f' (growth rate) on a common redshift grid,
        and 'k', 'pk' (linear matter power spectrum at z = 0, k in units of 1/chi, covering [1e-4, kmax]).
    b : callable
        Linear bias b(z).
    nz : callable
        Mean density nz(z) per unit redshift and steradian (only its shape matters here).
    zbins : (nbin + 1,) array
        Redshift bin edges.
    nomega : int
        Number of radial frequencies per bin.
    lmax : int
        Maximum multipole; use 3 nside - 1 of the mask for mean and covariance.
    ells : sorted int array, optional
        Multipoles at which the theory is computed (default: all); linearly interpolated in between, and zero outside
        [min(ells), max(ells)]. A subset such as np.r_[0:50, 50:lmax:4, lmax] is much faster.
    kmax : float, optional
        Maximum wavenumber, in units of 1/chi.
    nk : int, optional
        Number of wavenumbers in (0, kmax].
    dchi : float, optional
        Radial integration step, in units of chi.

    Returns
    -------
    G, Gt : (nbin, nomega, nbin, nomega, lmax + 1) complex arrays
        G[b, i, c, j, l] = <a_lm(b, omega_i) a_lm(c, omega_j)^*>, Gt[b, i, c, j, l] = <a_lm(b, omega_i) a_lm(c, -omega_j)^*>.
    """
    k = np.linspace(1e-4, kmax, nk)
    pk = np.exp(np.interp(np.log(k), np.log(cosmo['k']), np.log(cosmo['pk'])))
    k_weight = np.sqrt(2 / np.pi * k**2 * pk * trapz_weights(k))[:, None]                 # G = sum_k Delta_i Delta_j^* k_weight^2

    def at(z, key):
        return np.interp(z, cosmo['z'], cosmo[key])
    kernels = []                                                                         # per redshift bin
    for z0, z1 in zip(zbins[:-1], zbins[1:]):
        z = np.linspace(z0, z1, int((at(z1, 'chi') - at(z0, 'chi')) / dchi) + 2)
        n = nz(z)
        # radial weights dz nbar exp(-i omega (z - z0)) / sqrt(int dz nbar^2), as in the estimator
        dz = trapz_weights(z)
        radial = np.exp(-1j * np.pi * np.outer(np.arange(nomega), z - z0) / (z1 - z0)) * dz * n / np.sqrt(dz @ n**2)
        kernels.append((np.outer(k, at(z, 'chi')), b(z) * at(z, 'D'), at(z, 'f') * at(z, 'D'), radial))
    ells = np.arange(lmax + 1) if ells is None else np.asarray(ells)
    x = np.arange(0, kmax * at(zbins[-1], 'chi') + 2e-3, 1e-3)                             # tabulation grid of j_l(x), j_l''(x)
    G = np.zeros((2, len(zbins) - 1, nomega, len(zbins) - 1, nomega, len(ells)), complex)
    for il, l in enumerate(ells):
        jl = [spherical_jn(max(n, 0), x) for n in (l - 2, l, l + 2)]
        d2jl = (l * (l - 1) * jl[0] / (2 * l - 1) - (l**2 / (2 * l - 1) + (l + 1)**2 / (2 * l + 3)) * jl[1]
                + (l + 1) * (l + 2) * jl[2] / (2 * l + 3)) / (2 * l + 1)                  # j_l'' from the recurrence relations
        # Delta_i(k) = int dz (radial weight) [b D j_l(k chi) - f D j_l''(k chi)]   (density + Kaiser term)
        delta = k_weight * np.concatenate([(bD * np.interp(kchi, x, jl[1]) - fD * np.interp(kchi, x, d2jl)) @ radial.T
                                           for kchi, bD, fD, radial in kernels], 1)
        G[..., il] = np.array([delta.T @ delta.conj(), delta.T @ delta]).reshape(G.shape[:-1])
    G = interp1d(ells, G, axis=-1, bounds_error=False, fill_value=0)(np.arange(lmax + 1))
    return G[0], G[1]


def mean(G, Gt, win, width=1, pairs=False):
    """Expectation value of the measurement, i.e. the full-sky theory through the mask operator.

    Parameters
    ----------
    G, Gt : (nbin, nomega, nbin, nomega, 3 nside) complex arrays
        Full-sky theory from spectra().
    win : dict
        Mask operators from window().
    width, pairs : optional
        As in measure().

    Returns
    -------
    array
        Expectation of measure(..., width=width, pairs=pairs), with the same shape.
    """
    ibin = np.arange(len(G))
    return bin_omega(G[ibin, :, ibin] @ win['theory'].T, Gt[ibin, :, ibin] @ win['theory'].T, width, pairs)


def covariance(G, Gt, nz, zbins, win, width=1, pairs=False):
    """Gaussian covariance of the measurement, including the correlations between redshift bins.

    Computed in the improved narrow-kernel approximation (Garcia-Garcia et al. 2019, arXiv:1906.11765), with the Poisson noise
    of the given background density and the mask (and radial integral constraint) of `win`. The approximation is not guaranteed
    to be positive definite; check its eigenvalues before inverting very large matrices.

    Parameters
    ----------
    G, Gt : (nbin, nomega, nbin, nomega, 3 nside) complex arrays
        Full-sky theory from spectra().
    nz : callable
        Mean density nz(z) per unit redshift and steradian, n_g = nz(z) Omega(n) (sets the shot noise).
    zbins : (nbin + 1,) array
        Redshift bin edges.
    win : dict
        Mask operators from window().
    width, pairs : optional
        As in measure().

    Returns
    -------
    (ndata, ndata) array
        Covariance of measure(..., width=width, pairs=pairs).ravel(), ordered as (redshift bin, row, multipole band).
    """
    nbin, nomega, nell = G.shape[0], G.shape[1], G.shape[-1]
    mask_mean, mask2_mean, nband = np.mean(win['mask']), np.mean(win['mask']**2), len(win['binning'])
    band = bands(nomega, width if pairs else 1)                  # pairs: bands averaged before the Wick sums; Z(omega): after
    nB = len(band)
    average = np.kron(np.eye(nbin), band)                        # band averaging over (redshift bin, omega)

    # Poisson noise of <F_i F_j^*>, <F_i F_j> in the normalisation of G: <Omega> int dz nbar exp(-i (w_i -+ w_j)(z - z0)) / N
    noise = np.zeros((2, nbin * nomega, nbin * nomega), complex)
    for ibin, (z0, z1) in enumerate(zip(zbins[:-1], zbins[1:])):
        z = np.linspace(z0, z1, 10001)
        n = nz(z)
        phase = np.exp(-1j * np.pi * np.outer(np.arange(nomega), z - z0) / (z1 - z0))
        w = trapz_weights(z) * n * mask_mean / (mask2_mean * np.trapezoid(n**2, z))
        sl = slice(ibin * nomega, (ibin + 1) * nomega)
        noise[:, sl, sl] = (phase * w) @ phase.conj().T, (phase * w) @ phase.T

    # Spectra of the signed bands a = (+-, bin, A), with F_{-A} = F_A^*: <F_a F_b^*> = [[G, Gt], [Gt^*, G^*]], shape (ell, a, b).
    # Coupled through the mask (signal) plus noise times its shape, and divided by <Omega^2> (iNKA).
    def signed(g, gt):
        return np.block([[g, gt], [gt.conj(), g.conj()]])

    def band_average(x):
        return average @ x @ average.T
    S = signed(*(band_average(x.reshape(nbin * nomega, nbin * nomega, nell).transpose(2, 0, 1)) for x in (G, Gt)))
    S = (np.tensordot(win['coupling'] * win['pixwin2'], S, 1) / mask2_mean
         + win['noise_shape'][:, None, None] * signed(*map(band_average, noise))) / mask2_mean
    used = win['binning'].any(0)                                 # multipoles entering the bandpowers
    # DS[a, b, i, l] = binning[i, l] S_l(a, b); DSX = DS @ M2_ll' / (2l' + 1), the iNKA four-point coupling
    DS = win['binning'][:, used] * S[used].transpose(1, 2, 0)[:, :, None]
    DSX = DS @ (win['coupling2'] / (2 * np.arange(nell) + 1))[np.ix_(used, used)]

    # Each row is Re(c Z(a, b)) with c = 1 (real part) or -i (imaginary part), a = (+, bin, A), b = (+-, bin, B). By Wick,
    #   <Z(a, b) Z(c, d)^*> = S_ac S_bd^* + S_a,-d S_b,-c^*,   <Z(a, b) Z(c, d)> = S_ad S_bc^* + S_a,-c S_b,-d^*,
    #   Cov(Re c X, Re c' Y) = Re(c c'^* <X Y^*> + c c' <X Y>) / 2.
    labels = rows(nB, pairs)
    first = np.concatenate([ibin * nB + labels[:, 1] for ibin in range(nbin)])
    second = np.concatenate([(labels[:, 0] * nbin + ibin) * nB + labels[:, 2] for ibin in range(nbin)])
    part = np.tile(np.where(labels[:, 3], -1j, 1), nbin)[:, None, None]

    def flip(a):                                                 # index of the opposite-sign band
        return (a + nbin * nB) % (2 * nbin * nB)

    def herm(x):
        return x.conj().swapaxes(-1, -2)
    unique_pairs, pair_of_row = np.unique(np.c_[first, second], axis=0, return_inverse=True)
    cov = np.zeros((len(first), len(first), nband, nband))
    for ip, (a, b) in enumerate(unique_pairs):
        XYc = DSX[a, first] @ herm(DS[b, second]) + DSX[a, flip(second)] @ herm(DS[b, flip(first)])
        XY = DSX[a, second] @ herm(DS[b, first]) + DSX[a, flip(first)] @ herm(DS[b, flip(second)])
        for r in np.flatnonzero(pair_of_row == ip):
            cov[r] = 0.5 * np.real(part[r] * part.conj() * XYc + part[r] * part * XY)
    cov = cov.transpose(0, 2, 1, 3).reshape(len(first) * nband, -1)
    if not pairs:                                                # average Z(omega) over the frequency bands
        P = np.kron(np.kron(np.eye(nbin), bands(nomega, width)), np.eye(nband))
        cov = P @ cov @ P.T
    return 0.5 * (cov + cov.T)
