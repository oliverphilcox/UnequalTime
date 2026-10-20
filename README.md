# UnequalTime
UnequalTime is a Python code that estimates the three-dimensional angular power spectrum, $\mathcal{Z}_\ell(\omega)$, of galaxy surveys. This is the radial Fourier transform of the unequal time correlation function $C_\ell(z,z')$, and is computed directly from a catalog of angular positions and redshifts, without a fiducial cosmology. The code also computes the linear theory model and the Gaussian covariance, and is built on [NaMaster](https://github.com/LSSTDESC/NaMaster) and [healpy](https://github.com/healpy/healpy).

In each redshift bin, we form the frequency-space density field
$$a_{\ell m}(\omega) = \frac{1}{\mathcal{N}^{1/2}}\int d\hat{\mathbf{n}}\,Y^*_{\ell m}(\hat{\mathbf{n}})\int dz\,e^{-i\omega z}[n_g(\hat{\mathbf{n}},z)-\bar n_g(\hat{\mathbf{n}},z)]w(z)$$
where $n_g$ is the galaxy number density, $\bar n_g(\hat{\mathbf{n}},z) = \bar n(z)\Omega(\hat{\mathbf{n}})$ is the background density for angular mask $\Omega$, $w(z)$ is a radial weight, and $\mathcal{N}$ is a normalization. The power spectrum estimator is
$$\widehat{\mathcal{Z}}_{\ell}(\omega,\omega') = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell}a_{\ell m}(\omega)a^*_{\ell m}(\omega') - \mathcal{Z}_\ell^{\rm shot}(\omega,\omega')$$
with $\omega$ in multiples of the fundamental frequency $\pi/\Delta z$ for a redshift bin of width $\Delta z$. The main output is the diagonal spectrum, $\widehat{\mathcal{Z}}_\ell(\omega)\equiv\widehat{\mathcal{Z}}_\ell(\omega,\omega)$, but the off-diagonal spectra can also be computed. The code accounts for general (non-binary) angular masks, the pixel window function, and the radial integral constraint arising from estimating $\bar n(z)$ from the survey itself.

UnequalTime contains the following modules:
- `estimator`: mask operators and the $\widehat{\mathcal{Z}}_\ell(\omega,\omega')$ estimator
- `theory`: linear theory (non-Limber, with Kaiser redshift-space distortions) and the Gaussian covariance

The basic usage is the following:
```
# Import code
import unequaltime as ut
import numpy as np

# Compute the mask operators
win = ut.window(mask, # angular mask (HEALPix map)
                ell_edges, # ell-bin edges
                ric=True) # if True, nbar(z) is estimated from the data, giving a radial integral constraint

# Compute the power spectrum
Zl = ut.measure(ra, dec, z, # galaxy positions (in degrees) and redshifts
                win, # mask operators
                zbins, # redshift-bin edges
                nomega, # number of frequencies in each redshift bin
                width=1, # number of frequencies in each omega-bin
                pairs=False, # if True, compute all Z_ell(omega, omega'), not just Z_ell(omega)
                w=None) # radial weights, w(z)

# Compute the linear theory
G, Gt = ut.spectra(cosmo, # dictionary of redshift-dependent z, chi, D, f, and linear power spectrum k, pk
                   b, # linear bias, b(z)
                   nz, # background density, nbar(z)
                   zbins, nomega,
                   lmax) # maximum multipole (3*nside-1)
Zl_theory = ut.mean(G, Gt, win) # include mask and binning

# Compute the Gaussian covariance
cov = ut.covariance(G, Gt, nz, zbins, win)
```
The power spectrum has shape `(nbin, nomega, nell)`, *i.e.* redshift-bin, frequency and $\ell$-bin. If $\bar n(z)$ is known, use `ric=False` in `window` and pass `nz` to `measure`. Each function is documented in its docstring.

Further details are described in the tutorial:
- [Demo](examples/demo.ipynb): computing and fitting $\mathcal{Z}_\ell(\omega)$ for a test galaxy catalog, generated with [make_catalog.py](examples/make_catalog.py)

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Stanford)

## Dependencies
- Python 3.10+
- `numpy` (2.0+), `scipy`
- `healpy`
- `pymaster` [NaMaster, 2.0+]
- `matplotlib` [for the tutorial]

## Installation
### Install with pip
```bash
pip install "unequaltime @ git+https://github.com/oliverphilcox/UnequalTime"
```

### Install from source
```bash
git clone https://github.com/oliverphilcox/UnequalTime.git
cd UnequalTime
pip install -e .
```

## References
**Code references:**
1. Alonso, D., Sanchez, J., Slosar, A., "A unified pseudo-$C_\ell$ framework", (2019) ([arXiv](https://arxiv.org/abs/1809.09603))
2. García-García, C., Alonso, D., Bellini, E., "Disconnected pseudo-$C_\ell$ covariances for projected large-scale structure data", (2019) ([arXiv](https://arxiv.org/abs/1906.11765))
3. Barnett, A. H., Magland, J., af Klinteberg, L., "A Parallel Nonuniform Fast Fourier Transform Library Based on an 'Exponential of Semicircle' Kernel", (2019) ([arXiv](https://arxiv.org/abs/1808.06736))

## License
MIT; see [LICENSE](LICENSE).
