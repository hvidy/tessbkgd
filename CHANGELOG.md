# Changelog

All notable changes to `tessbkgd` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-14

### Added
- **Functional `lightkurve` API (`subtract_background`, `estimate_background`)**: Pure functional entry points operating across all raw rows of `hdu[1]` without mutating caller `TargetPixelFile` instances, returning `BackgroundEstimate` and optional `BackgroundDiagnostics`.
- **2D QR-equilibrated Chebyshev Huber–Tukey IRLS solver (`solver.py`)**: Batched JAX solver executing 4 convex Huber steps (`c_H = 1.345`) plus 1 rank-guarded Tukey biweight step (`c_B = 4.685`) with adaptive MAD standardization over 8,192-cadence chunks (`< 270 MB` RAM) on an orthonormalized Chebyshev basis (`equilibrate_chebyshev_basis`).
- **Analytical sandwich covariance & column-coherent `SMEAR_ERR` propagation**: Evaluates Huber–Street–Carroll–Ruppert sandwich Cholesky factors (`BKG_CHOL`, storing pixel-independent sky covariance conditional on column smear for `halophot`) and propagates off-diagonal column-coherent `SMEAR_ERR` sky variance into standalone `FLUX_BKG_ERR` while preserving idempotent `subtract_background` round-trips.
- **`HALOCVER = 1` cross-package FITS contract (`_contract.py`)**: Standardized `HALOCVER`, `PixelQuality` (`NONE`, `STRAP_COL`, `SAT_BLEED`, `SMEAR_UNRESOLVED`, `SOURCE_MASK`, `BKG_USED`), `BkgCovariance`, `read_bkgsub_state`, `read_smear_err`, `read_bkg_covariance`, `write_tessbkgd_provenance`, `pack_bkg_chol`, `unpack_bkg_chol`, and `resolve_cache_dir` (with `$TESSBKGD_CACHE_DIR` -> `$TESS_BRIGHT_CACHE_DIR` -> `$SCRATCH/.cache/tess_bright` -> `$XDG_CACHE_HOME/tess_bright` -> `~/.cache/tess_bright` precedence, `create` flag, and symlink-aware `/fred` inode protection).
- **Dark-cadence detrended sky masking & WCS Gaia source masks (`masking.py`)**: Gradient-tolerant dark-25% median background masking (`build_background_mask`), per-cadence detrended temporal MAD variable-star rejection, circular Moffat-wing catalogue exclusion (`build_wcs_source_mask`), and curvature/rank auto-downgrade down to `BKG_ORD = 0` (including tiny `2x2` cutouts).
- **Locally scoped 64-bit JAX precision helpers (`_precision.py`)**: `@in_float64`, `run_in`, and `cast_tree` executing numerical kernels in `float64` without setting global `jax_enable_x64` on import.

### Changed
- Modernized package layout to `src/tessbkgd` with `hatchling` (`python >= 3.11`, `astropy >= 6.0`, `lightkurve >= 2.4`, `jax >= 0.4.30`) and excluded notebooks and FITS artefacts from source distributions (`sdist`).

### Deprecated
- Legacy `bkgd_tpf` subclass (`get_bkgd`, `subtract_bkgd`): retained as a compatibility shim delegating onto the v2.0 solver and emitting `DeprecationWarning`.
