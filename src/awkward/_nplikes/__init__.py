# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward._nplikes.cupy
import awkward._nplikes.jax
import awkward._nplikes.numpy
import awkward._nplikes.typetracer
import awkward._nplikes.virtual
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awkward._nplikes.numpy_like import NumpyLike
from awkward._nplikes.array_like import ArrayLike


def to_nplike(
    array: ArrayLike, nplike: NumpyLike, *, from_nplike: NumpyLike | None = None
) -> ArrayLike:
    if from_nplike is None:
        from_nplike = nplike_of_obj(array, default=None)
        if from_nplike is None:
            raise TypeError(
                f"internal error: expected an array supported by an existing nplike, got {type(array).__name__!r}"
            )

    if from_nplike is to_nplike:
        return array

    if nplike.known_data and not from_nplike.known_data:
        raise TypeError(
            "Converting from an nplike without known data to an nplike with known data is not supported"
        )

    # Copy to host memory
    if isinstance(from_nplike, awkward._nplikes.cupy.Cupy) and not isinstance(
        nplike, awkward._nplikes.cupy.Cupy
    ):
        array = array.get()  # type: ignore[attr-defined]

    return nplike.asarray(array)
