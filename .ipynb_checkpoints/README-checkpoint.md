# UnequalTime

Unequal-time angular power spectra of a galaxy catalogue,
$\mathcal Z_\ell(\omega,\omega')$: the angular cross-spectrum of $\int_{z_0}^{z_0+\Delta z} dz\, e^{-i\omega (z-z_0)}\,\delta n_g(\hat n, z)$
at radial frequencies $\omega, \omega'$, with $\omega_m = \pi m/\Delta z$ ($m <$ `nw`) in each redshift bin.
It handles general masks and the radial integral constraint. Theory: linear, non-Limber, Kaiser RSD. Covariance: Gaussian.

```bash
pip install .        # numpy, scipy, healpy, pymaster
```

```python
import numpy as np, unequaltime as ut

win = ut.window(mask, ell_edges)                             # mask: HEALPix map of the angular selection, in [0, 1]
Zpp, Zpm = ut.measure(ra, dec, z, win, zbins, nw)            # Z(w_i, w_j), Z(w_i, -w_j): (nbin, nw, nw, nband)
Z = np.einsum('bjjl->bjl', Zpp).real                         # Z_ell(w): (nbin, nw, nband)
X, rows = ut.bin_omega(Zpp, Zpm, width=4)                    # w-band pairs (kind, A, B, Re/Im): (nbin, nrow, nband)

G, Gt = ut.spectra(cosmo, b, nz, zbins, nw, lmax=3 * nside - 1)
X_th = ut.bin_omega(*ut.mean(G, Gt, win), width=4)[0]
cov = ut.covariance(G, Gt, nz, zbins, win, width=4)          # of X.ravel()
```

- `cosmo`: dict of arrays `z, chi, D, f` (with D(0) = 1) and `k, pk` (linear power spectrum at z = 0); `b(z)` is the linear bias and `nz(z)` = dN/dz dΩ per steradian.
- By default the background n(z) is estimated from the catalogue, and the radial integral constraint this introduces is modelled. For a known n(z), use `window(mask, ell_edges, ric=False)` and `measure(..., nz=nz)`.
