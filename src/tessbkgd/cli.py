"""Command-line interface for ``tessbkgd`` background subtraction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import lightkurve as lk

from . import __version__
from .masking import DEFAULT_EXCLUSION_RADIUS, OrderSpec, format_order
from .plotting import plot_diagnostics
from .tessbkgd import subtract_background


def _parse_cli_order(raw: str | None) -> OrderSpec:
    if raw is None:
        return None
    s = raw.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Expected 'oy,ox' pair, got {raw!r}.")
        return (int(parts[0]), int(parts[1]))
    try:
        return int(s)
    except ValueError:
        return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tessbkgd",
        description="Fit and subtract a 2D Chebyshev background from a TESS TPF.",
    )
    p.add_argument("input", type=Path, help="Input TESS TPF (.fits).")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output FITS path (default: <stem>_bkgsub.fits).",
    )
    p.add_argument(
        "--order", type=_parse_cli_order, default=None,
        help="Order: 0..3, name ('linear'/'quadratic'), or 'oy,ox' (e.g. '2,1').",
    )
    p.add_argument(
        "--exclusion-radius", type=float, default=DEFAULT_EXCLUSION_RADIUS,
        help=f"Exclusion radius in px (default: {DEFAULT_EXCLUSION_RADIUS:g}).",
    )
    p.add_argument(
        "--no-downgrade", action="store_true", help="Disable auto-downgrade."
    )
    p.add_argument("--plot", type=Path, default=None, help="Save 4-panel figure path.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output FITS.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print fit progress.")
    p.add_argument("--version", action="version", version=f"tessbkgd {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_path: Path = args.input
    default_out = in_path.with_name(f"{in_path.stem}_bkgsub{in_path.suffix}")
    out_path = args.output or default_out
    clean, diag = subtract_background(
        lk.read(str(in_path)),
        order=args.order,
        exclusion_radius=args.exclusion_radius,
        allow_downgrade=not args.no_downgrade,
        return_diagnostics=True,
        verbose=args.verbose,
    )
    clean.to_fits(str(out_path), overwrite=args.overwrite)
    if args.plot is not None:
        fig, _ = plot_diagnostics(clean, diag, save_path=str(args.plot))
        import matplotlib.pyplot as plt

        plt.close(fig)
    if args.verbose:
        print(
            f"Wrote {out_path} (order={format_order(diag.effective_order)}, "
            f"n_bg={int(diag.bkg_mask.sum())}, r={diag.target_flux_corr:+.3f})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
