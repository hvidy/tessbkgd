"""Spatially and temporally varying background estimation for TESS TPFs."""

from ._contract import PixelQuality
from .plotting import plot_diagnostics
from .tessbkgd import (
    BackgroundConfig,
    BackgroundDiagnostics,
    BackgroundEstimate,
    bkgd_tpf,
    estimate_background,
    subtract_background,
)

__version__ = "2.0.1"

__all__ = [
    "BackgroundConfig",
    "BackgroundDiagnostics",
    "BackgroundEstimate",
    "PixelQuality",
    "__version__",
    "bkgd_tpf",
    "estimate_background",
    "plot_diagnostics",
    "subtract_background",
]
