"""Main interface: a survey (mask, binning, background density and weights) with its measurement, theory and covariance."""
import copy
import numpy as np
import healpy as hp
from . import estimator, theory
from .estimator import as_function


class Survey:
    """Unequal time power spectrum Z_ell(omega) or Z_ell(omega, omega') of a galaxy survey.

    The survey fixes the angular mask, the redshift, frequency and multipole binning, the background density nbar(z) and the
    radial weights w(z), which are then used consistently by measure(), theory(), expected() and covariance().

    Parameters
    ----------
    mask : (npix,) array
        HEALPix map (RING ordering) of the angular mask Omega >= 0, in any normalization. Multipoles run to 3 nside - 1.
    zbins : (nbin + 1,) array
        Redshift-bin edges.
    nomega : int
        Number of frequencies in each redshift bin, omega = m pi / (z1 - z0) for m = 0, ..., nomega - 1.
    ell_edges : (nell + 1,) int array
        Edges of the ell-bins [ell_i, ell_i+1), at most 3 nside.
    nbar : 'data', callable or float, optional
        Background density nbar(z) per unit redshift and steradian, with nbar_g(n, z) = nbar(z) Omega(n). 'data' (default)
        estimates it from each measured catalog, in shells of width Delta z / (nomega - 1), and includes the resulting radial
        integral constraint.
    w : callable, float or None, optional
        Radial weight w(z) (default 1).
    width : int, optional
        Number of frequencies in each omega-bin: bin 0 is omega = 0 alone, then `width` consecutive frequencies each.
    pairs : bool, optional
        Use all pairs Z_ell(omega_A, +-omega_B) of omega-bins, not just Z_ell(omega).
    verbose : bool, optional
        Print progress.

    Attributes
    ----------
    zbins, nomega, width, pairs, w, verbose : the inputs (w as a function).
    omega : (nbin, n_omega_bins) array
        Centers of the omega-bins in each redshift bin.
    ell : (nell,) array
        Centers of the ell-bins.
    lmax : int
        Maximum multipole of the theory, 3 nside - 1.
    nbar : callable or None
        Background density; with nbar='data', that of the last measured catalog (None before the first measurement). Shared
        with the copies made by rebin().
    rows : (nrow, 4) int array or None
        With pairs=True, the labels (kind, A, B, part) of the rows of vector() (see estimator.rows).
    window : dict
        Mask operators (see estimator.window).
    """

    def __init__(self, mask, zbins, nomega, ell_edges, nbar='data', w=None, width=1, pairs=False, verbose=True):
        if isinstance(nbar, str) and nbar != 'data':
            raise ValueError("nbar must be 'data', a function nbar(z) or a constant")
        mask, ell_edges = np.asarray(mask, dtype=float), np.asarray(ell_edges)
        self.zbins, self.nomega, self.verbose = np.asarray(zbins, dtype=float), nomega, verbose
        self.from_data = isinstance(nbar, str)
        self._state = dict(nbar=None if self.from_data else as_function(nbar))   # shared with the copies made by rebin()
        self.w = as_function(1. if w is None else w)
        self.lmax = 3 * hp.npix2nside(len(mask)) - 1
        self.window = estimator.window(mask, ell_edges, ric=self.from_data, verbose=verbose)
        self.ell = (ell_edges[1:] + ell_edges[:-1] - 1) / 2
        self.width, self.pairs = width, pairs
        self._set_bins()

    @property
    def nbar(self):
        return self._state['nbar']

    def _set_bins(self):
        frequencies = np.pi * np.arange(self.nomega) / np.diff(self.zbins)[:, None]
        self.omega = frequencies @ estimator.omega_bins(self.nomega, self.width).T
        self.rows = estimator.rows(self.omega.shape[1]) if self.pairs else None

    def rebin(self, width=None, pairs=None):
        """Copy of the survey with a different omega-binning (sharing the mask operators and nbar, so cheap)."""
        new = copy.copy(self)
        new.width = self.width if width is None else width
        new.pairs = self.pairs if pairs is None else pairs
        new._set_bins()
        return new

    def measure(self, ra, dec, z):
        """Measure the power spectrum of a catalog.

        Parameters
        ----------
        ra, dec : (ngal,) arrays
            Angular positions in degrees. Galaxies in pixels where the mask vanishes are ignored.
        z : (ngal,) array
            Redshifts.

        Returns
        -------
        array or tuple of arrays
            pairs=False: (nbin, n_omega_bins, nell) array, Z_ell(omega).
            pairs=True: two (nbin, n_omega_bins, n_omega_bins, nell) complex arrays, Z_ell(omega_A, omega_B) and
            Z_ell(omega_A, -omega_B).
        """
        if self.from_data:
            self._state['nbar'] = estimator.background(ra, dec, z, self.window, self.zbins, self.nomega)
        Z_plus, Z_minus = estimator.measure(ra, dec, z, self.window, self.zbins, self.nomega,
                                            nbar=None if self.from_data else self.nbar, w=self.w, verbose=self.verbose)
        return estimator.bin_omega(Z_plus, Z_minus, self.width, self.pairs)

    def theory(self, cl, b, kmax=0.2, ells=None, cross_bins=True, nk=1500, dchi=0.2):
        """Unmasked linear theory of all frequency pairs (non-Limber, Kaiser redshift-space distortions).

        Parameters
        ----------
        cl : classy.Class
            Computed CLASS instance of the cosmology, with output 'mPk' and P_k_max_1/Mpc >= h kmax.
        b : callable or float
            Linear bias b(z).
        kmax : float, optional
            Maximum wavenumber [h/Mpc].
        ells : sorted int array, optional
            Multipoles to compute, interpolated in between and zero outside [min(ells), max(ells)] (default: all below 50, then
            every fourth, and lmax).
        cross_bins : bool, optional
            Whether to compute the spectra between redshift bins (needed for the covariance, not for expected()).
        nk, dchi : optional
            Number of wavenumbers, and radial integration step [Mpc/h].

        Returns
        -------
        dict
            Z_plus, Z_minus: (nbin, nomega, nbin, nomega, 3 nside) complex arrays, Z_ell(omega_i, omega_j) and Z_ell(omega_i,
            -omega_j) between redshift bins p and q (index [p, i, q, j, ell]); nbar: the background density used; cross_bins.
        """
        if self.nbar is None:
            raise ValueError("with nbar='data', measure() a catalog first: the theory uses its nbar(z)")
        cosmo = theory.cosmology(cl, self.zbins[-1], kmax)
        Z_plus, Z_minus = theory.spectra(cosmo, as_function(b), self.nbar, self.zbins, self.nomega, self.lmax, w=self.w,
                                         ells=ells, kmax=kmax, nk=nk, dchi=dchi, cross_bins=cross_bins, verbose=self.verbose)
        return dict(Z_plus=Z_plus, Z_minus=Z_minus, nbar=self.nbar, cross_bins=cross_bins)

    def expected(self, spectra):
        """Expectation of measure() for the theory spectra, including the mask, the radial integral constraint and the binning.

        Parameters
        ----------
        spectra : dict
            Output of theory().

        Returns
        -------
        array or tuple of arrays
            As measure().
        """
        ibin, operator = np.arange(len(self.zbins) - 1), self.window['theory'].T
        return estimator.bin_omega(spectra['Z_plus'][ibin, :, ibin] @ operator, spectra['Z_minus'][ibin, :, ibin] @ operator,
                                   self.width, self.pairs)

    def vector(self, Z):
        """Flat real data vector of a measurement or expectation (for pairs=True, the independent pairs, labelled by rows).

        Parameters
        ----------
        Z : array or tuple of arrays
            Output of measure() or expected().

        Returns
        -------
        (ndata,) array
            Data vector, ordered as (redshift bin, omega-bin or row, ell-bin).
        """
        return (estimator.pair_vector(*Z) if self.pairs else Z).ravel()

    def covariance(self, spectra):
        """Gaussian covariance of vector(measure(...)), including correlations between redshift bins.

        Parameters
        ----------
        spectra : dict
            Output of theory() with cross_bins=True.

        Returns
        -------
        (ndata, ndata) array
            Covariance (improved narrow-kernel approximation, with the shot noise of spectra['nbar']).
        """
        if not spectra['cross_bins'] and len(self.zbins) > 2:
            raise ValueError('the covariance needs the spectra between redshift bins: use theory(..., cross_bins=True)')
        return theory.covariance(spectra['Z_plus'], spectra['Z_minus'], spectra['nbar'], self.zbins, self.window, self.width,
                                 self.pairs, self.w, verbose=self.verbose)
