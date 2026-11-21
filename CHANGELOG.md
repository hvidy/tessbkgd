# Changelog

All notable changes to `tessbkgd` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-11-20

### Fixed
- **Single-pixel-per-column `SMEAR_ERR` fallback (`solver.py`, `[TBKG-01]`)**: In `_compute_smear_decomposed_chunk`, fall back to 2D polynomial fit residuals across active pixels (with finite degrees-of-freedom factor `n_act / max(n_act - p_dim, 1)`) whenever every active column has at most 1 inlier (`n_el == 0`) or vertical degrees of freedom vanish, eliminating `100% NaN` `BKG_CHOL` / `FLUX_BKG_ERR` when `FLUX_ERR` is absent and restoring continuous Street–Carroll–Ruppert `c_rob` rescaling when `FLUX_ERR` is present.
- **Catalogue proper-motion epoch propagation (`masking.py`, `tessbkgd.py`, `[TBKG-02]`)**: Propagate `pmra` (`pm_ra_cosdec`), `pmdec`, and `ref_epoch` (defaulting to `2016.0` for Gaia tables containing `source_id` / `phot_g_mean_mag`, else `2000.0`) to the TPF mid-sector observation epoch `obs_time` (inferred from `TIME` + `BJDREFI`/`BJDREFF`, `tpf.time`, or `MJD-OBS`/`DATE-OBS`, suppressing `ErfaWarning` when distance is absent) before WCS projection in `build_wcs_source_mask`.
- **Stale covariance invalidation & strict `HALOCVER` validation (`tessbkgd.py`, `_contract.py`, `[TBKG-03b]`)**: Strip stale `BKG_CHOL` and `BKG_BASIS` extensions (emitting a `UserWarning`) when legacy `bkgd_tpf.flux_bkg = ...` overwrites `FLUX_BKG` directly, and validate `HDU[0]` and `HDU[1]` `HALOCVER` cards individually so boolean cards (`True`) cannot bypass integer checks via `1 == True`.
- **Wide interior bleed-gap curvature guard (`masking.py`, `[TBKG-P1]`)**: Detect interior all-masked column or row gaps spanning `> 50%` of the stamp width or height in `apply_curvature_and_rank_guards` (and coarse surface initialization) and downgrade `order >= 2` to `linear` (`1`) when `allow_downgrade=True`, preventing quadratic arch extrapolation of flank stellar wings across saturated cores.

### Documentation
- **Cross-package `HALOCVER = 1` smear covariance semantics (`[XPKG-01]`)**: Documented explicitly in `solver.py` and `_contract.py` that `HALOCVER = 1` schema is unchanged — `BKG_CHOL` intentionally represents pixel-independent background fit covariance conditional on column smear, while `BKG_BASIS` together with `PIXEL_QUALITY & BKG_USED` carry the WLS design geometry for downstream oblique smear propagation in `halophot`.

## [2.0.0] - 2026-11-14

### Added
- **Functional `lightkurve` API (`subtract_background`, `estimate_background`)**: Pure functional entry points operating across all raw rows of `hdu[1]` without mutating caller `TargetPixelFile` instances, returning `BackgroundEstimate` and optional `BackgroundDiagnostics`.
- **2D QR-equilibrated Chebyshev Huber–Tukey IRLS solver (`solver.py`)**: Batched JAX solver executing 4 convex Huber steps (`c_H = 1.345`) plus 1 rank-guarded Tukey biweight step (`c_B = 4.685`) with adaptive MAD standardization over 8,192-cadence chunks (`< 270 MB` RAM) on an orthonormalized Chebyshev basis (`equilibrate_chebyshev_basis`).
- **Analytical sandwich covariance & column-coherent `SMEAR_ERR` propagation**: Separates `P x P` coefficient covariance into pixel-independent sandwich Cholesky factors (`BKG_CHOL`, conditional on column smear with Street–Carroll–Ruppert scaling on within-column centred residuals for `halophot`) and positive semi-definite column-coherent `SMEAR_ERR` covariance (`Sigma_{beta,smear,t}`) propagated into standalone `FLUX_BKG_ERR`, while preserving idempotent `subtract_background` round-trips.
- **`HALOCVER = 1` cross-package FITS contract (`_contract.py`)**: Standardized `HALOCVER`, `PixelQuality` (`NONE`, `STRAP_COL`, `SAT_BLEED`, `SMEAR_UNRESOLVED`, `SOURCE_MASK`, `BKG_USED`), `BkgCovariance`, `read_bkgsub_state`, `read_smear_err`, `read_bkg_covariance`, `write_tessbkgd_provenance`, `pack_bkg_chol`, `unpack_bkg_chol`, and `resolve_cache_dir` (with `$TESSBKGD_CACHE_DIR` -> `$TESS_BRIGHT_CACHE_DIR` -> `$SCRATCH/.cache/tess_bright` -> `$XDG_CACHE_HOME/tess_bright` -> `~/.cache/tess_bright` precedence, `create` flag, and symlink-aware `/fred` inode protection).
- **Dark-cadence detrended sky masking & WCS Gaia source masks (`masking.py`)**: Gradient-tolerant dark-25% median background masking (`build_background_mask`), per-cadence detrended temporal MAD variable-star rejection, circular Moffat-wing catalogue exclusion (`build_wcs_source_mask`), and curvature/rank auto-downgrade down to `BKG_ORD = 0` (including tiny `2x2` cutouts).
- **Locally scoped 64-bit JAX precision helpers (`_precision.py`)**: `@in_float64`, `run_in`, and `cast_tree` executing numerical kernels in `float64` without setting global `jax_enable_x64` on import.

### Changed
- Modernized package layout to `src/tessbkgd` with `hatchling` (`python >= 3.11`, `astropy >= 6.0`, `lightkurve >= 2.4`, `jax >= 0.4.30`) and excluded notebooks and FITS artefacts from source distributions (`sdist`).

### Deprecated
- Legacy `bkgd_tpf` subclass (`get_bkgd`, `subtract_bkgd`): retained as a compatibility shim delegating onto the v2.0 solver and emitting `DeprecationWarning`.
