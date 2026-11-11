"""Locally scoped 64-bit precision helpers for numerical entry points.

``tessbkgd`` never enables ``jax_enable_x64`` globally on import: numerical
entry points execute inside :func:`run_in` (a local ``jax.enable_x64`` context)
after :func:`cast_tree`, restoring the caller's prior JAX precision state on
exit.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

F = TypeVar("F", bound=Callable[..., object])

_enable_x64 = getattr(jax, "enable_x64", None)
if _enable_x64 is None:  # pragma: no cover - JAX < 0.8 fallback
    from jax.experimental import enable_x64 as _enable_x64

_DTYPES = {
    "float32": (jnp.float32, jnp.complex64, np.float32, np.complex64),
    "float64": (jnp.float64, jnp.complex128, np.float64, np.complex128),
}


def _check_dtype(dtype: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(
            f"dtype must be 'float32' or 'float64', not {dtype!r}."
        )


@contextlib.contextmanager
def run_in(dtype: str) -> Iterator[None]:
    """Execute the enclosed block with JAX 64-bit mode enabled or disabled.

    Parameters
    ----------
    dtype : {"float32", "float64"}
        Target floating-point precision. Prior JAX precision state is restored
        on exit.
    """
    _check_dtype(dtype)
    with _enable_x64(dtype == "float64"):
        yield


def cast_tree(tree: object, dtype: str = "float64") -> object:
    """Cast floating and complex arrays in a PyTree to ``dtype``.

    Integer, boolean, and non-array leaves pass through unchanged. Astropy
    ``Quantity`` leaves (subclasses of ``numpy.ndarray`` carrying ``.unit``)
    retain their units and NumPy container while casting floating values.
    """
    _check_dtype(dtype)
    j_real, j_cplx, np_real, np_cplx = _DTYPES[dtype]

    def _cast_leaf(leaf: object) -> object:
        if isinstance(leaf, np.ndarray) and hasattr(leaf, "unit"):
            if np.issubdtype(leaf.dtype, np.complexfloating):
                return leaf.astype(np_cplx)
            if np.issubdtype(leaf.dtype, np.floating):
                return leaf.astype(np_real)
            return leaf
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            return leaf
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
            return jnp.asarray(leaf, dtype=j_cplx)
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return jnp.asarray(leaf, dtype=j_real)
        return leaf

    return jax.tree_util.tree_map(_cast_leaf, tree)


def in_float64(fn: F) -> F:
    """Run ``fn`` inside ``jax.enable_x64(True)`` with floating inputs cast to float64.

    Restores the prior JAX ``x64`` configuration state on exit. Non-array
    arguments and iterators pass through unchanged.
    """

    def _cast_arg(arg: object) -> object:
        return arg if isinstance(arg, Iterator) else cast_tree(arg, "float64")

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        with run_in("float64"):
            cast_args = [_cast_arg(a) for a in args]
            cast_kwargs = {k: _cast_arg(v) for k, v in kwargs.items()}
            return fn(*cast_args, **cast_kwargs)

    return wrapped  # type: ignore[return-value]
