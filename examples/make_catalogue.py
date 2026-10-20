"""Generate the demo catalogue: Poisson galaxies in a Gaussian, linear, redshift-space (Kaiser) density field on the sphere.

The density on fine redshift shells is drawn in harmonic space from the same linear theory as unequaltime.spectra,
    delta_lm(z) = sum_k sqrt(w_k) [b D j_l(k chi) - f D j_l''(k chi)] g_klm,   w_k = (2/pi) k^2 P(k) dk,   g_klm unit Gaussian,
so that the catalogue has exactly the spectra predicted by unequaltime (with the same k grid). The amplitude is low enough
(k < 0.05 h/Mpc) that a realisation with 1 + delta > 0 everywhere exists: the first such seed is used (seed 0 for the shipped
data), so the galaxies are an exact Poisson sample of nbar(z) Omega(n) (1 + delta), without clipping.

Run from this directory: python make_catalogue.py  (writes data/catalogue.npz; a few minutes).
"""
import numpy as np
import healpy as hp
from scipy.special import spherical_jn

KMAX, NK = 0.05, 1500                                       # wavenumbers of the field [h/Mpc]; as unequaltime.spectra(kmax=KMAX)
ZMIN, ZMAX, DZ = 0.1, 0.7, 5e-4                             # redshift range and shell width
bias = lambda z: 0.7 + 0.5 * z                              # linear bias
nz = lambda z: 1.5e5 * (z / 0.4)**2 * np.exp(1 - (z / 0.4)**2)    # mean galaxies per unit redshift and steradian at Omega = 1

cosmo = dict(np.load('data/cosmology.npz'))
mask = hp.read_map('data/mask.fits')
nside = hp.npix2nside(len(mask))
lmax = 3 * nside - 1
at = lambda z, key: np.interp(z, cosmo['z'], cosmo[key])
zs = np.arange(ZMIN + DZ / 2, ZMAX, DZ)                     # shell centres
k = np.linspace(1e-4, KMAX, NK)
pk = np.exp(np.interp(np.log(k), np.log(cosmo['k']), np.log(cosmo['pk'])))
wk = 2 / np.pi * k**2 * pk * np.r_[0.5, np.ones(NK - 2), 0.5] * (k[1] - k[0])    # trapezoid rule, as unequaltime.spectra
kchi = np.outer(k, at(zs, 'chi'))                                               # (k, shell)
bD, fD = bias(zs) * at(zs, 'D'), at(zs, 'f') * at(zs, 'D')
x = np.arange(0, kchi.max() + 2e-3, 1e-3)
ell, em = hp.Alm.getlm(lmax)


def transfer(l):
    """T_l(k, shell) = sqrt(w_k) [b D j_l(k chi) - f D j_l''(k chi)]."""
    jl = [spherical_jn(max(n, 0), x) for n in (l - 2, l, l + 2)]
    d2jl = (l * (l - 1) * jl[0] / (2 * l - 1) - (l**2 / (2 * l - 1) + (l + 1)**2 / (2 * l + 3)) * jl[1]
            + (l + 1) * (l + 2) * jl[2] / (2 * l + 3)) / (2 * l + 1)
    return np.sqrt(wk)[:, None] * (bD * np.interp(kchi, x, jl[1]) - fD * np.interp(kchi, x, d2jl))


for seed in range(100):
    rng = np.random.default_rng(seed)
    alm = np.zeros((len(zs), len(ell)), complex)
    for l in range(lmax + 1):
        idx = np.flatnonzero(ell == l)
        g = rng.normal(size=(NK, len(idx))) + 1j * rng.normal(size=(NK, len(idx)))
        g = np.where(em[idx] == 0, np.sqrt(2) * g.real, g) / np.sqrt(2)   # unit variance; real for m = 0
        alm[:, idx] = transfer(l).T @ g
    delta = np.array([hp.alm2map(a, nside, pixwin=True) for a in alm])  # (shell, pixel), pixel-averaged
    dmin = delta[:, mask > 0].min()
    print(f'seed {seed}: min delta = {dmin:.3f}', flush=True)
    if dmin > -1:
        break
else:
    raise RuntimeError('no realisation with 1 + delta > 0 everywhere; lower the clustering amplitude')

# Poisson sample of nbar(z) Omega(n) (1 + delta) by thinning uniform points on the sphere in each shell
dmax = delta.max()
nprop = rng.poisson(4 * np.pi * nz(zs) * DZ * mask.max() * (1 + dmax))
shell = np.repeat(np.arange(len(zs)), nprop)
u = rng.uniform(-1, 1, len(shell))
ra, dec = rng.uniform(0, 360, len(shell)), np.degrees(np.arcsin(u))
pix = hp.ang2pix(nside, ra, dec, lonlat=True)
keep = rng.random(len(shell)) < mask[pix] * (1 + delta[shell, pix]) / (mask.max() * (1 + dmax))
z = zs[shell] + rng.uniform(-DZ / 2, DZ / 2, len(shell))
np.savez_compressed('data/catalogue.npz', ra=ra[keep].astype(np.float32), dec=dec[keep].astype(np.float32),
                    z=z[keep].astype(np.float32))
print(f'{keep.sum()} galaxies written to data/catalogue.npz')
