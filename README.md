# UnequalTime

**Unequal-time angular power spectra of galaxy surveys.**

`unequaltime` measures the angular power spectra $\mathcal Z_\ell(\omega)$ and $\mathcal Z_\ell(\omega,\omega')$ of a galaxy
catalogue Fourier transformed along redshift, and provides their linear theory and Gaussian covariance. It works directly on a
(RA, Dec, z) catalogue and a HEALPix mask, and is built on [NaMaster](https://github.com/LSSTDESC/NaMaster) and
[healpy](https://github.com/healpy/healpy).

## Overview

Angular power spectra of thin redshift slices, $C_\ell(z, z')$, contain the full radial information of a survey but are
numerous and strongly correlated. `unequaltime` instead Fourier transforms the density along redshift within each redshift bin
$[z_0, z_0+\Delta z)$,

$$F_\omega(\hat n) = \frac{1}{\sqrt N}\int_{z_0}^{z_0+\Delta z} dz\; e^{-i\omega(z-z_0)}\,\big[n_g(\hat n, z) - \bar n_g(\hat n, z)\big],
\qquad N = \int dz \int \frac{d\hat n}{4\pi}\,\bar n_g^2 ,$$

where $n_g$ is the number of galaxies per unit redshift and solid angle, and measures the angular power spectra of these maps,

$$\mathcal Z_\ell(\omega, \omega') = \langle F_{\ell m}(\omega)\,F^*_{\ell m}(\omega') \rangle, \qquad \mathcal Z_\ell(\omega) \equiv \mathcal Z_\ell(\omega,\omega).$$

The frequencies are $\omega_m = \pi m/\Delta z$, $m = 0, 1, \dots,$ `nomega` $- 1$; mode $m$ has a radial wavelength of
$2\Delta z/m$ in redshift. $\mathcal Z_\ell(0)$ is proportional to the angular power spectrum of the bin,
$\mathcal Z_\ell(0) = C_\ell^{\rm bin} \big(\int dz\, \bar n\big)^2 / \int dz\, \bar n^2$, and higher frequencies isolate progressively
shorter radial scales. $\mathcal Z_\ell$ has units of redshift and does not depend on the overall number density.
The main product is $\mathcal Z_\ell(\omega)$; the spectra of all frequency pairs, $\mathcal Z_\ell(\omega, \pm\omega')$, are available as an option.

### Features
- **Estimator** from a catalogue: frequency maps by a non-uniform FFT along redshift, harmonic transforms with healpy, binned
  pseudo-$C_\ell$ with exact Poisson shot-noise subtraction and pixel-window correction.
- **General masks**: any angular selection $\Omega(\hat n) \geq 0$ (binary or not); the mode coupling (from NaMaster) is applied to the theory.
- **Background density** either estimated from the catalogue itself, with the resulting radial integral constraint modelled
  exactly, or specified.
- **Frequency binning**: $\mathcal Z_\ell(\omega)$ averaged in bands of $\omega$, or the cross-spectra of all pairs of bands.
- **Theory**: linear, non-Limber, with Kaiser redshift-space distortions and arbitrary bias $b(z)$, growth $D(z)$ and growth rate $f(z)$.
- **Covariance**: Gaussian, including shot noise, the mask, and correlations between frequencies and redshift bins.

## Installation

```bash
git clone https://github.com/oliverphilcox/UnequalTime.git
cd UnequalTime
pip install -e .                 # or: pip install -e ".[examples,test]" for the notebook and tests
```

Requirements: Python ≥ 3.10, numpy ≥ 2, scipy, healpy, pymaster ≥ 2. Run the tests with `pytest tests`.

## Quick start

```python
import numpy as np
import healpy as hp
import unequaltime as ut

ra, dec, z = ...                             # galaxy positions (degrees) and redshifts
mask = hp.read_map('mask.fits')              # angular selection (HEALPix, RING ordering)

zbins = np.array([0.1, 0.4, 0.7])            # redshift bin edges
nomega = 31                                  # radial frequencies per bin
ell_edges = np.arange(2, 250, 8)             # bandpowers [ell_i, ell_{i+1}); at most 3 nside

win = ut.window(mask, ell_edges)             # mask operators (computed once per mask)
Z = ut.measure(ra, dec, z, win, zbins, nomega)   # Z_ell(omega), shape (nbin, nomega, nband)

omega = np.pi * np.arange(nomega) / np.diff(zbins)[:, None]   # frequencies of each bin, (nbin, nomega)
ell = (ell_edges[1:] + ell_edges[:-1] - 1) / 2                 # band centres, nband = len(ell_edges) - 1
```

`Z[b, m, i]` is $\mathcal Z_\ell(\omega_m)$ in redshift bin `b` and multipole band `i`.

### Frequency bands and pairs

The same two options apply to `measure`, `mean` and `covariance`:

| option | output of `measure` |
|---|---|
| default | $\mathcal Z_\ell(\omega_m)$: `(nbin, nomega, nband)` |
| `width=4` | $\mathcal Z_\ell(\omega)$ averaged over bands of 4 frequencies: `(nbin, n_omega_bands, nband)` |
| `pairs=True` | cross-spectra of all pairs of frequency bands, real and imaginary parts: `(nbin, nrow, nband)` |

The frequency bands are band 0 $= \{\omega_0 = 0\}$, then `width` consecutive frequencies each (the last may be narrower), so
`n_omega_bands = len(ut.bands(nomega, width))`, and `ut.bands(nomega, width) @ omega[b]` gives the band centres.

With `pairs=True`, $\mathcal Z(A, \pm B)$ is the cross-spectrum of the band-averaged maps of bands $A$ and $B$ (the mean of
$\mathcal Z(\omega, \pm\omega')$ over $\omega \in A$, $\omega' \in B$). The rows are labelled by `ut.rows(n_omega_bands)`, each
`(kind, A, B, part)`: `kind` 0 for $\mathcal Z(A, +B)$ with $A \leq B$, `kind` 1 for $\mathcal Z(A, -B)$ with $0 < A \leq B$, and
`part` 0 / 1 for the real / imaginary part. These are all the independent pairs: $\mathcal Z(\omega,\omega') = \mathcal Z(\omega',\omega)^*$,
$\mathcal Z(0, -\omega) = \mathcal Z(0, \omega)^*$ (the $\omega = 0$ map is real), and $\mathcal Z(A, +A)$ is real.

`.ravel()` orders any output as (redshift bin, row, multipole band), as does the covariance.

### Theory and covariance

The theory needs three inputs:
- `nz(z)`: mean number of galaxies per unit redshift and steradian, so that $\bar n_g(\hat n, z) =$ `nz(z)` $\times$ `mask`$(\hat n)$
  (the mask may have any normalisation). For example, from the catalogue:
  ```python
  counts, z_edges = np.histogram(z, 100, (zbins[0], zbins[-1]))
  area = mask.sum() * hp.nside2pixarea(hp.npix2nside(len(mask)))           # int dn Omega(n)
  nz = lambda x: np.interp(x, (z_edges[1:] + z_edges[:-1]) / 2, counts / np.diff(z_edges) / area)
  ```
- `b(z)`: linear galaxy bias.
- `cosmo`: dictionary of arrays `z`, `chi` (comoving distance), `D` (linear growth factor, $D(0) = 1$) and `f` (growth rate) on
  a redshift grid, and `k`, `pk` (linear matter power spectrum at $z = 0$, with $k$ in units of 1/`chi`, covering
  $10^{-4} \leq k \leq$ `kmax`). For example, with [CLASS](https://github.com/lesgourg/class_public) (distances in Mpc):
  ```python
  from classy import Class
  c = Class()
  c.set({'output': 'mPk', 'z_max_pk': 3, 'P_k_max_1/Mpc': 2})
  c.compute()
  zz, kk = np.linspace(0, 3, 3001), np.logspace(-4, 0, 500)
  cosmo = dict(z=zz, chi=np.array([c.comoving_distance(x) for x in zz]),
               D=np.array([c.scale_independent_growth_factor(x) for x in zz]),
               f=np.array([c.scale_independent_growth_factor_f(x) for x in zz]),
               k=kk, pk=np.array([c.pk_lin(x, 0) for x in kk]))
  ```

Then

```python
lmax = 3 * hp.npix2nside(len(mask)) - 1
G, Gt = ut.spectra(cosmo, b, nz, zbins, nomega, lmax)    # full-sky theory of all frequency pairs
Z_th = ut.mean(G, Gt, win)                               # expectation of Z, same shape
cov = ut.covariance(G, Gt, nz, zbins, win)               # covariance of Z.ravel()
```

`spectra` evaluates every multipole up to `lmax` by default; pass e.g. `ells=np.r_[0:50, 50:lmax:4, lmax]` to compute a subset and
interpolate (much faster). `kmax` (default 0.2, in units of 1/`chi`) sets the smallest scale of the linear theory, and `dchi`
(default 0.2, in units of `chi`) the radial integration step.

### Known background density

By default the background $n(z)$ is estimated from the catalogue. Each map then has its masked mean removed, a radial integral
constraint that is included exactly in the window, the shot noise and the normalisation. If $n(z)$ is known in advance, pass it
and build the window without the constraint:

```python
win = ut.window(mask, ell_edges, ric=False)
Z = ut.measure(ra, dec, z, win, zbins, nomega, nz=nz)
```

## Example: fitting the clustering amplitude

```python
win = ut.window(mask, ell_edges)
Z = ut.measure(ra, dec, z, win, zbins, nomega, width=4).ravel()

G, Gt = ut.spectra(cosmo, b, nz, zbins, nomega, 3 * hp.npix2nside(len(mask)) - 1)
Z_th = ut.mean(G, Gt, win, width=4).ravel()
icov = np.linalg.inv(ut.covariance(G, Gt, nz, zbins, win, width=4))

A = (Z_th @ icov @ Z) / (Z_th @ icov @ Z_th)             # best-fit amplitude of the model
sigma_A = (Z_th @ icov @ Z_th)**-0.5                     # its 1-sigma error
chi2 = (Z - A * Z_th) @ icov @ (Z - A * Z_th)            # goodness of fit (len(Z) - 1 degrees of freedom)
```

Passing `pairs=True` to `measure`, `mean` and `covariance` uses the cross-spectra of all frequency bands instead.

## Demo

[`examples/demo.ipynb`](examples/demo.ipynb) runs the full pipeline on a small test dataset in `examples/data/`, with plots:
- `catalogue.npz`: 355,000 galaxies at $0.1 < z < 0.7$, Poisson-sampled from a Gaussian linear density field with bias
  $b(z) = 0.9 + 0.5z$ and Kaiser redshift-space distortions on $k < 0.05\,h/{\rm Mpc}$, generated by
  [`examples/make_catalogue.py`](examples/make_catalogue.py) from the same linear theory as `spectra`;
- `mask.fits`: a footprint with varying completeness, at nside 64;
- `cosmology.npz`: the `cosmo` table (flat ΛCDM with $h = 0.6766$, $\omega_b = 0.02242$, $\omega_{\rm cdm} = 0.11933$,
  $n_s = 0.9665$, $\ln 10^{10}A_s = 3.047$; distances in Mpc/$h$).

## API

| function | description |
|---|---|
| `window(mask, ell_edges, ric=True)` | mask operators: mode coupling, binning, pixel window, shot-noise shape |
| `measure(ra, dec, z, win, zbins, nomega, width=1, pairs=False, nz=None, dz=0.005)` | $\mathcal Z_\ell(\omega)$, or all band pairs, from a catalogue |
| `spectra(cosmo, b, nz, zbins, nomega, lmax, ells=None, kmax=0.2, nk=1500, dchi=0.2)` | full-sky linear theory of all frequency pairs |
| `mean(G, Gt, win, width=1, pairs=False)` | expectation of `measure` |
| `covariance(G, Gt, nz, zbins, win, width=1, pairs=False)` | Gaussian covariance of `measure(...).ravel()` |
| `bin_omega(Zpp, Zpm, width=1, pairs=False)` | frequency binning of the spectra of all frequency pairs |
| `bands(nomega, width)`, `rows(n_omega_bands)` | frequency-band averaging matrix; labels of the pair data vector |

Each function is documented in its docstring, e.g. `help(ut.measure)`.

## Notes
- Phases are measured from the start of each redshift bin, $e^{-i\omega(z-z_0)}$.
- The measured spectra are coupled pseudo-$C_\ell$ binned in $\ell$, with the shot noise subtracted and the pixel window removed;
  the mask is not deconvolved, but applied to the theory by `mean` (and to the covariance).
- The covariance uses the improved narrow-kernel approximation, accurate to a few per cent for typical masks; it is not
  guaranteed to be positive definite, so check its eigenvalues before inverting very large matrices.
- Cost, for nside 256, 2 redshift bins, `nomega = 61` and 2 million galaxies: `window` 1–4 min (the integral constraint needs
  3 nside pairs of harmonic transforms), `measure` 30 s and 4 GB, `spectra` 3 min at 100 multipoles with the default `kmax`,
  `covariance` 1 min for $\mathcal Z_\ell(\omega)$ and 5–10 min and 16 GB for all pairs with `width=4`. The pair covariance scales as
  the square of the data-vector length.

## References
- NaMaster: Alonso, Sanchez & Slosar (2019), [arXiv:1809.09603](https://arxiv.org/abs/1809.09603).
- Pseudo-$C_\ell$ covariance (improved narrow-kernel approximation): García-García, Alonso & Bellini (2019), [arXiv:1906.11765](https://arxiv.org/abs/1906.11765).
- Non-uniform FFT (exponential-of-semicircle kernel): Barnett, Magland & af Klinteberg (2019), [arXiv:1808.06736](https://arxiv.org/abs/1808.06736).

## License

MIT; see [LICENSE](LICENSE).
