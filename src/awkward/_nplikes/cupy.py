# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy

import awkward as ak
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpy_like import ArrayLike
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual
from awkward._typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@register_nplike
class Cupy(ArrayModuleNumpyLike):
    is_eager: Final = False
    supports_structured_dtypes: Final = False

    def __init__(self):
        import awkward._connect.cuda  # noqa: F401

        self._module = ak._connect.cuda.import_cupy("Awkward Arrays with CUDA")

    @property
    def ma(self):
        raise ValueError(
            "CUDA arrays cannot have missing values until CuPy implements "
            "numpy.ma.MaskedArray"
        )

    @property
    def char(self):
        raise ValueError(
            "CUDA arrays cannot do string manipulations until CuPy implements "
            "numpy.char"
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def frombuffer(
        self, buffer, *, dtype: DTypeLike | None = None, count: ShapeItem = -1
    ) -> ArrayLike:
        assert not isinstance(buffer, (PlaceholderArray, VirtualArray))
        assert not isinstance(count, (PlaceholderArray, VirtualArray))
        np_array = numpy.frombuffer(buffer, dtype=dtype, count=count)
        return self._module.asarray(np_array)

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> bool:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        if x1.shape != x2.shape:
            return False
        else:
            return self._module.array_equal(x1, x2, equal_nan=equal_nan).get()

    def repeat(
        self, x: ArrayLike, repeats: ArrayLike | int, *, axis: int | None = None
    ):
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(repeats, PlaceholderArray)
        x, repeats = materialize_if_virtual(x, repeats)
        if axis is not None:
            raise NotImplementedError(f"repeat for CuPy with axis={axis!r}")
        # https://github.com/cupy/cupy/issues/3849
        if isinstance(repeats, self._module.ndarray):
            all_stops = self._module.cumsum(repeats)
            parents = self._module.zeros(all_stops[-1].item(), dtype=int)
            stops, stop_counts = self._module.unique(all_stops[:-1], return_counts=True)
            parents[stops] = stop_counts
            self._module.cumsum(parents, out=parents)
            return x[parents]
        else:
            return self._module.repeat(x, repeats=repeats)

    # For all reducers: https://github.com/cupy/cupy/issues/3819

    def all(
        self,
        x: ArrayLike,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        out = self._module.all(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def any(
        self,
        x: ArrayLike,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        out = self._module.any(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def count_nonzero(
        self, x: ArrayLike, *, axis: ShapeItem | tuple[ShapeItem, ...] | None = None
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        assert isinstance(axis, int) or axis is None
        out = self._module.count_nonzero(x, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def min(
        self,
        x: ArrayLike,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        out = self._module.min(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def max(
        self,
        x: ArrayLike,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        out = self._module.max(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    @classmethod
    def is_own_array_type(cls, type_: type) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a cupy buffer, otherwise `False`.

        """
        module, _, suffix = type_.__module__.partition(".")
        return module == "cupy"

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        # TODO: Should this materialize virtual arrays?
        if isinstance(x, (PlaceholderArray, VirtualArray)):
            return True
        else:
            return x.flags["C_CONTIGUOUS"]  # type: ignore[attr-defined]
