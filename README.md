# UnequalTime
UnequalTime is a Python code that estimates the unequal time power spectrum, $\mathcal{Z}_\ell(\omega)$ or $\mathcal{Z}_\ell(\omega,\omega')$, for galaxy surveys. This is the radial Fourier transform of the unequal time correlation function $C_\ell(z,z')$ in $z-z'$, and is computed directly from a catalog of angular positions and redshifts, without a fiducial cosmology. The code also computes the linear theory model and the Gaussian covariance, and is built on [NaMaster](https://github.com/LSSTDESC/NaMaster) and [healpy](https://github.com/healpy/healpy).

In each redshift bin, we form the frequency-space density field

$$a_{\ell m}(\omega) = \frac{1}{\mathcal{N}^{1/2}}\int d\hat{\mathbf{n}}\,Y^*_{\ell m}(\hat{\mathbf{n}})\int dz\,e^{-i\omega z}[n_g(\hat{\mathbf{n}},z)-\bar n_g(\hat{\mathbf{n}},z)]w(z)$$

where $n_g$ is the galaxy number density, $\bar n_g(\hat{\mathbf{n}},z) = \bar n(z)\Omega(\hat{\mathbf{n}})$ is the background density for angular mask $\Omega$, $w(z)$ is a radial weight, and $\mathcal{N}$ is a normalization. This is computed using a spherical harmonic transform in angular space and a non-uniform fast Fourier transform in radial space. The power spectrum estimator is

$$\widehat{\mathcal{Z}}_{\ell}(\omega,\omega') = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell}a_{\ell m}(\omega)a^*_{\ell m}(\omega') - \mathcal{Z}_\ell^{\rm shot}(\omega,\omega')$$

with $\omega$ in multiples of the fundamental frequency $\pi/\Delta z$ for a redshift bin of width $\Delta z$. The main output is the diagonal spectrum, $\widehat{\mathcal{Z}}_\ell(\omega)\equiv\widehat{\mathcal{Z}}_\ell(\omega,\omega)$ (which typically dominates), but the off-diagonal spectra can also be computed. The code accounts for general angular masks, the pixel window function, and the radial integral constraint arising from estimating $\bar n(z)$ from the survey itself.

UnequalTime contains the following modules:
- `survey`: definitions of the mask, binning, background density and weights
- `estimator`: estimation of $\widehat{\mathcal{Z}}_\ell(\omega,\omega')$ from a (RA, Dec, z) dataset
- `theory`: linear theory (non-Limber, including wide-angle effects and Kaiser redshift-space distortions) and the Gaussian covariance

The basic usage is the following:
```
# Import code
import unequaltime as ut
import numpy as np

# Define the survey
survey = ut.Survey(mask, # angular mask (HEALPix map)
                   zbins, # redshift-bin edges
                   nomega, # number of frequencies in each redshift bin
                   ell_edges, # ell-bin edges
                   nbar='data', # background density nbar(z): 'data' (estimated from the catalog) or a function
                   w=None, # radial weights w(z)
                   width=1, # number of frequencies in each omega-bin
                   pairs=False) # if True, use all Z_ell(omega, omega'), not just Z_ell(omega)

# Compute the power spectrum
Zl_est = survey.measure(ra, dec, z) # galaxy positions (in degrees) and redshifts

# Compute the linear theory, Z_l(omega, omega') and Z_l(omega, -omega')
spectra = survey.theory(cl, # computed CLASS instance (output 'mPk')
                        b) # linear bias, b(z)
Zl_theory = survey.expected(spectra) # include mask and binning

# Compute the Gaussian covariance
cov = survey.covariance(spectra)
```
If `pairs=False`, `Zl_est` and `Zl_theory` have shape `(nbin_z, nbin_omega, nbin_ell)`, *i.e.* redshift-bin, frequency and $\ell$-bin, with bin centers `survey.omega` and `survey.ell`. If `pairs=True`, we return the (binned) complex power spectra $\mathcal{Z}_\ell(\omega,\omega')$ and $\mathcal{Z}_\ell(\omega,-\omega')$ with dimensions `(nbin_z, nbin_omega, nbin_omega, nbin_ell)`. Note that the radial density, $\bar n(z)$, can either be given explicitly or estimated from the data by specifying `nbar='data'`; in the latter case, the theoretical model includes radial integral constraint corrections. For convenience, we also include the function `survey.vector(Zl)`, which flattens the data vector, keeping only the independent pairs; this matches the format of the covariance, `cov`.

Further details are described in the [tutorial](examples/demo.ipynb).

## Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Stanford)

## Dependencies
- Python 3.10+
- `numpy` (2.0+), `scipy`
- `healpy`
- `pymaster` [NaMaster, 2.0+]
- `classy` [CLASS, for theory models]
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
