# UnequalTime

Unequal-time angular power spectra of a galaxy catalogue,
$\mathcal Z_\ell(\omega,\omega')$: the angular cross-spectrum of $\int dz\, e^{-i\omega z}\,\delta_g(\hat n, z)$
at radial frequencies $\omega$ and $\omega'$, with $\omega_m = \pi m/\Delta z$ in each redshift bin of width $\Delta z$.
It handles general masks and the radial integral constraint. Theory: linear, non-Limber, Kaiser RSD. Covariance: Gaussian.

```bash
pip install .        # numpy, scipy, healpy, pymaster
```

```python
import unequaltime as ut

win = ut.window(mask, ell_edges)                             # mask: HEALPix map of the angular selection, in [0, 1]
Zpp, Zpm, N = ut.measure(ra, dec, z, mask, zbins, nw, win)   # Z(w_i, w_j), Z(w_i, -w_j): (nbin, nw, nw, nband)
Z = Zpp.diagonal(0, 1, 2).real                               # Z_ell(w): (nbin, nband, nw)
X, rows = ut.bin_omega(Zpp, Zpm, width=4)                    # w-band pairs (kind, A, B, Re/Im): (nbin, nrow, nband)

G, Gt = ut.theory(cosmo, b, nz, zbins, nw, lmax=len(win['b2']) - 1)
Zpp_th, Zpm_th = ut.mean(G, Gt, win)
cov = ut.covariance(G, Gt, nz, zbins, win, width=4)          # of X.ravel()
```

- `cosmo`: dict of arrays `z, chi, D, f` (with D(0) = 1) and `k, pk` (linear power spectrum at z = 0); `b(z)` is the linear bias and `nz(z)` = dN/dz dΩ per steradian.
- By default the background n(z) is estimated from the catalogue, and the radial integral constraint this introduces is modelled. For a known n(z), use `window(mask, ell_edges, ric=False)` and `measure(..., nz=nz)`.
