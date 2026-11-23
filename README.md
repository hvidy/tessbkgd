# tessbkgd

[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/tessbkgd.svg)](https://badge.fury.io/py/tessbkgd)

Spatially and temporally varying sky background estimation for *TESS* Target Pixel Files (TPFs), Full-Frame Image (FFI) cutouts, and raw 3D flux cubes, engineered to interoperate cleanly with `lightkurve`, `mundey`, and `halophot` under the `HALOCVER = 1` FITS contract.

## Features (v2)

- **Tunable Saturated-Star Exclusion Zones & Dark-Cadence Masking**: Excludes `PixelQuality` flags (`STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK`), dead constant-zero edges (without biasing zero-mean or negative sky), and a user-tunable circular `exclusion_radius` (default `15.0` px) around saturated star cores (`>= 7.5e4 e-/s` or `SAT_BLEED`) plus a `±2`-column lateral bleed buffer. Supports optional WCS `(ra, dec, mag)` catalogues (`catalog_table`, with `gaia_sources` alias) with automatic proper-motion propagation, and emits a `UserWarning` whenever fitted background correlates strongly with target aperture flux (`|r| > 0.5`) or fewer than `20` sky pixels survive.
- **Anisotropic 2D Chebyshev Basis & Per-Axis Curvature Guards**: Supports `order` as an integer (`0..3`), string (`'pedestal'`, `'linear'`, `'quadratic'`, `'cubic'`), or `(order_y, order_x)` pair. When `order=None`, defaults independently per axis (`2 if axis >= 40 else 1`), so tall bright-star strips (e.g. `1287x11`, `135x25`) default to `(2, 1)` (quadratic vertical sky profile, linear horizontal tilt). Wide interior gaps (`> 50%`) and rank guards downgrade per axis and always emit a `UserWarning` when `effective_order != requested_order`.
- **Median-Seeded `@in_float64` JAX Huber–Tukey IRLS & Exact HC2 Sandwich Covariance**: Seeds IRLS from the spatial median pedestal (immune to high-leverage corner outlier clusters), runs 8 convex Huber steps ($c_H = 1.345$) plus 2 curvature-guarded Tukey biweight steps ($c_B = 4.685$) in bounded-RAM cadence chunks, and computes the exact $P \times P$ HC2 Huber–Street–Carroll–Ruppert matrix sandwich covariance with $(1 - h_{ii})^{-1}$ leverage scaling.
- **Visual Diagnostics & CLI**: Built-in 4-panel `plot_diagnostics(tpf, result)` (and `diag.plot(tpf)`) plus the `tessbkgd` command-line interface.

## Installation

Requires Python `>= 3.11`:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import lightkurve as lk
import tessbkgd

tpf = lk.read("tess...-s_tp.fits")

# Fit and subtract an anisotropic 2D Chebyshev background without mutating tpf:
clean_tpf, diag = tessbkgd.subtract_background(
    tpf,
    order=None,            # auto per axis: (2, 1) on tall strips, (2, 2) on >=40x40
    exclusion_radius=15.0, # circular exclusion around saturated cores
    catalog_table=None,    # optional Astropy Table / dict / (N, 3) array (alias: gaia_sources)
    return_diagnostics=True,
)

# Inspect 4-panel diagnostics (median image + mask, model, RdBu_r z-score residuals, r):
fig, axes = diag.plot(tpf, save_path="bkg_diagnostics.png")

# Estimate background on a TPF or a raw 3D NumPy array (n_cad, ny, nx):
est = tessbkgd.estimate_background(tpf)
est_raw = tessbkgd.estimate_background(flux_cube_3d, order=(2, 1))
```

### Command-Line Interface

```bash
tessbkgd input_tpf.fits -o clean_tpf.fits --order 2,1 --exclusion-radius 15 --plot diag.png --overwrite
```

### Legacy `bkgd_tpf` Shim (Deprecated)

`tessbkgd.bkgd_tpf` remains available as a deprecated compatibility subclass delegating onto the v2 solver and emitting `DeprecationWarning`:

```python
tpf.__class__ = tessbkgd.bkgd_tpf
bkgd, bkgd_mask = tpf.subtract_bkgd(order="linear")
```

## Public API (`tessbkgd.__all__`)

`tessbkgd` exports a concise 9-symbol public namespace:

- `subtract_background`, `estimate_background`, `plot_diagnostics`
- `BackgroundConfig`, `BackgroundEstimate`, `BackgroundDiagnostics`, `PixelQuality`
- `bkgd_tpf` (deprecated), `__version__`

Low-level `HALOCVER = 1` FITS contract helpers (`BkgCovariance`, `read_bkgsub_state`, `read_smear_err`, `read_bkg_covariance`, `write_tessbkgd_provenance`, `pack_bkg_chol`, `unpack_bkg_chol`, `resolve_cache_dir`) and local 64-bit JAX helpers (`in_float64`, `run_in`, `cast_tree`) live in `tessbkgd._contract` and `tessbkgd._precision`.
