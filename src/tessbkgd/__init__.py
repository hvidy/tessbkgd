"""Spatially and temporally varying background estimation for TESS TPFs."""

from . import masking, solver, tessbkgd
from ._contract import (
    HALOCVER,
    PixelQuality,
    pack_bkg_chol,
    read_bkgsub_state,
    resolve_cache_dir,
    unpack_bkg_chol,
    write_tessbkgd_provenance,
)
from ._precision import cast_tree, in_float64, run_in
from .masking import (
    ORDER_NAMES,
    BackgroundMaskResult,
    apply_curvature_and_rank_guards,
    build_background_mask,
    build_wcs_source_mask,
    n_basis_for_order,
    normalized_grid,
    parse_order,
    raw_chebyshev_2d,
)
from .solver import (
    SolverOutput,
    equilibrate_chebyshev_basis,
    reconstruct_background_chunked,
    solve_chebyshev_background,
)
from .tessbkgd import (
    BackgroundDiagnostics,
    BackgroundEstimate,
    bkgd_tpf,
    estimate_background,
    subtract_background,
)

name = "tessbkgd"
__version__ = "2.0.0.dev0"

__all__ = [
    "BackgroundDiagnostics",
    "BackgroundEstimate",
    "BackgroundMaskResult",
    "HALOCVER",
    "ORDER_NAMES",
    "PixelQuality",
    "SolverOutput",
    "__version__",
    "apply_curvature_and_rank_guards",
    "bkgd_tpf",
    "build_background_mask",
    "build_wcs_source_mask",
    "cast_tree",
    "equilibrate_chebyshev_basis",
    "estimate_background",
    "in_float64",
    "masking",
    "n_basis_for_order",
    "name",
    "normalized_grid",
    "pack_bkg_chol",
    "parse_order",
    "raw_chebyshev_2d",
    "read_bkgsub_state",
    "reconstruct_background_chunked",
    "resolve_cache_dir",
    "run_in",
    "solve_chebyshev_background",
    "solver",
    "subtract_background",
    "tessbkgd",
    "unpack_bkg_chol",
    "write_tessbkgd_provenance",
]
