# tessbkgd

[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/tessbkgd.svg)](https://badge.fury.io/py/tessbkgd)

Spatially and temporally varying sky background estimation for *TESS* Target Pixel Files (TPFs) and Full-Frame Image (FFI) cutouts, engineered to interoperate cleanly with `lightkurve`, `mundey`, and `halophot` under the `HALOCVER = 1` FITS contract.

## Installation

Requires Python `>= 3.11`:

```bash
pip install -e ".[dev]"
```

## Quick Start (`bkgd_tpf` Shim)

```python
import lightkurve as lk
import tessbkgd

tpf = lk.read("tess...-s_tp.fits")
tpf.__class__ = tessbkgd.bkgd_tpf

# Fit and subtract a 2D linear or quadratic background while honouring BKGSUB
# provenance across MAST SPOC, TessCut, and mundey v2.0 TPFs:
bkgd, bkgd_mask = tpf.subtract_bkgd(order="linear", verbose=True)
```

## Public Contract & Precision Helpers (`HALOCVER = 1`)

- `HALOCVER` (`1`), `PixelQuality` (`STRAP_COL`, `SAT_BLEED`, `SMEAR_UNRESOLVED`, `SOURCE_MASK`, `BKG_USED`), `read_bkgsub_state`, `write_tessbkgd_provenance`, `pack_bkg_chol`, `unpack_bkg_chol`: versioned FITS provenance and lockstep per-cadence `BKG_CHOL` uncertainty storage inside `hdu[1]` (`PIXELS`).
- `resolve_cache_dir`: symlink-aware cache path resolver defaulting to `$SCRATCH/.cache/tess_bright` and guarding `/fred/` inode quotas on Swinburne Ngarrgu Tindebeek / OzSTAR.
- `in_float64`, `run_in`, `cast_tree`: locally scoped 64-bit JAX precision helpers with zero global `jax_enable_x64` import side effects.
