# tessbkgd

[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/tessbkgd.svg)](https://badge.fury.io/py/tessbkgd)

Spatially and temporally varying sky background estimation for *TESS* Target Pixel Files (TPFs) and Full-Frame Image (FFI) cutouts, engineered to interoperate cleanly with `lightkurve`, `mundey`, and `halophot` under the `HALOCVER = 1` FITS contract.

## Features (v2.0)

- **Gradient-Tolerant Dark-Cadence Sky Masking**: Excludes `PixelQuality` flags (`STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK`) and dead $U \le 0$ edges, forms a dark-25% cadence median image, fits a robust tile-anchored coarse background surface $P_0(r, c)$ with lower-half sky MAD clipping, applies optional WCS `(ra, dec, mag)` circular star masks (`gaia_sources`), and rejects variable contaminants via per-cadence detrended temporal MAD so sweeping Earthshine fans never self-mask.
- **QR-Equilibrated 2D Chebyshev Basis & Curvature Guards**: Defaults to harmonic planes (`order=1`, $\nabla^2 B = 0$) on stamps with $\min(n_x, n_y) < 40$ and `order=2` on wider stamps ($\ge 40 \times 40$), supporting `0/'pedestal'`, `1/'linear'`, `2/'quadratic'`, and `3/'cubic'` with automatic rank/curvature downgrade and unit-conditioned QR equilibration ($\kappa(\Phi_{\Omega_{\text{bg}}}) = 1$).
- **Batched `@in_float64` JAX Huber–Tukey IRLS & Sandwich Covariance**: Runs 4 convex Huber steps ($c_H = 1.345$) plus 1 rank-guarded Tukey biweight step ($c_B = 4.685$) with adaptive per-step MAD standardization across 8,192-cadence chunks (`< 270 MB` RAM), propagating exact Huber–Street–Carroll–Ruppert sandwich covariances into `FLUX_BKG_ERR`, `FLUX_ERR`, and `BKG_CHOL` without enabling global `jax_enable_x64` on import.

## Installation

Requires Python `>= 3.11`:

```bash
pip install -e ".[dev]"
```

## Quick Start (`subtract_background` & `estimate_background`)

```python
import lightkurve as lk
import tessbkgd

tpf = lk.read("tess...-s_tp.fits")

# Fit and subtract a 2D Chebyshev background across all raw rows of hdu[1],
# returning a clean lightkurve.TargetPixelFile with idempotent BKGSUB provenance:
clean_tpf = tessbkgd.subtract_background(tpf, order=None)

# Inspect detailed solver diagnostics or supply a WCS source catalogue:
clean_tpf, diag = tessbkgd.subtract_background(
    tpf,
    order="quadratic",
    gaia_sources=catalogue_table,
    return_diagnostics=True,
)

# Estimate background and 1-sigma marginal uncertainties without copying/modifying tpf:
est = tessbkgd.estimate_background(tpf)
# est.bkg, est.bkg_err (astropy.units.Quantity), est.mask (2D bool ndarray)
```

### Legacy `bkgd_tpf` Shim (Deprecated)

`tessbkgd.bkgd_tpf` remains available as a `@deprecated` compatibility subclass delegating onto the v2.0 solver and emitting `DeprecationWarning`:

```python
tpf.__class__ = tessbkgd.bkgd_tpf
bkgd, bkgd_mask = tpf.subtract_bkgd(order="linear")
```

## Public Contract & Precision Helpers (`HALOCVER = 1`)

- `HALOCVER` (`1`), `PixelQuality` (`NONE`, `STRAP_COL`, `SAT_BLEED`, `SMEAR_UNRESOLVED`, `SOURCE_MASK`, `BKG_USED`), `BkgCovariance`, `read_bkgsub_state`, `read_smear_err`, `read_bkg_covariance`, `write_tessbkgd_provenance`, `pack_bkg_chol`, `unpack_bkg_chol`: versioned FITS provenance (`BKGSUB = True`, synchronized `BACKAPP = True` when present, `BKG_BASIS` with `BKG_ORD`, `PIXEL_QUALITY`, pixel-independent sandwich `BKG_CHOL` inside `hdu[1]`, and column-coherent `SMEAR_ERR` propagation into `FLUX_BKG_ERR`).
- `resolve_cache_dir`: symlink-aware cache path resolver checking `$TESSBKGD_CACHE_DIR`, `$TESS_BRIGHT_CACHE_DIR`, `$SCRATCH/.cache/tess_bright`, `$XDG_CACHE_HOME/tess_bright`, and `~/.cache/tess_bright`, guarding `/fred/` inode quotas on Swinburne Ngarrgu Tindebeek / OzSTAR.
- `in_float64`, `run_in`, `cast_tree`: locally scoped 64-bit JAX precision helpers with zero global `jax_enable_x64` import side effects.
