"""Spatially and temporally varying background estimation for TESS TPFs."""

from . import tessbkgd
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
from .tessbkgd import bkgd_tpf

name = "tessbkgd"
__version__ = "2.0.0.dev0"

__all__ = [
    "HALOCVER",
    "PixelQuality",
    "__version__",
    "bkgd_tpf",
    "cast_tree",
    "in_float64",
    "name",
    "pack_bkg_chol",
    "read_bkgsub_state",
    "resolve_cache_dir",
    "run_in",
    "tessbkgd",
    "unpack_bkg_chol",
    "write_tessbkgd_provenance",
]
