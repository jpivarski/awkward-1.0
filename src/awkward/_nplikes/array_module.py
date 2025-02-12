# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import inspect
import math
from functools import lru_cache

from awkward._nplikes.numpy_like import (
    ArrayLike,
    IndexType,
    NumpyLike,
    NumpyMetadata,
    UfuncLike,
    UniqueAllResult,
)
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual
from awkward._typing import TYPE_CHECKING, Any, DType, Final, Literal, TypeVar, cast

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

np = NumpyMetadata.instance()


@lru_cache
def _nplike_concatenate_has_casting(module: Any) -> bool:
    x = module.zeros(2)
    try:
        module.concatenate((x, x), casting="same_kind")
    except TypeError:
        return False
    else:
        return True


ArrayLikeT = TypeVar("ArrayLikeT", bound=ArrayLike)


class ArrayModuleNumpyLike(NumpyLike[ArrayLikeT]):
    known_data: Final[bool] = True
    _module: Any

    def prepare_ufunc(self, ufunc: UfuncLike) -> UfuncLike:
        return ufunc

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> ArrayLikeT | PlaceholderArray | VirtualArray:
        if isinstance(obj, PlaceholderArray) or (
            isinstance(obj, VirtualArray) and not obj.is_materialized
        ):
            assert obj.dtype == dtype or dtype is None
            return obj
        if isinstance(obj, VirtualArray) and obj.is_materialized:
            assert obj.dtype == dtype or dtype is None
            obj = obj.materialize()
        if copy:
            return self._module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            return self._module.asarray(obj, dtype=dtype)
        else:
            if getattr(obj, "dtype", dtype) != dtype:
                raise ValueError(
                    "asarray was called with copy=False for an array of a different dtype"
                )
            else:
                return self._module.asarray(obj, dtype=dtype)

    def ascontiguousarray(
        self, x: ArrayLikeT | PlaceholderArray | VirtualArray
    ) -> ArrayLikeT | PlaceholderArray | VirtualArray:
        # TODO: Should this materialize virtual arrays?
        if isinstance(x, (PlaceholderArray, VirtualArray)):
            return x
        else:
            return self._module.ascontiguousarray(x)

    def frombuffer(
        self, buffer, *, dtype: DTypeLike | None = None, count: ShapeItem = -1
    ) -> ArrayLikeT:
        if isinstance(buffer, PlaceholderArray):
            raise TypeError("placeholder arrays are not supported in `frombuffer`")
        if isinstance(buffer, VirtualArray):
            raise TypeError("virtual arrays are not supported in `frombuffer`")
        return self._module.frombuffer(buffer, dtype=dtype, count=count)

    def from_dlpack(self, x: Any) -> ArrayLikeT:
        return self._module.from_dlpack(x)

    def zeros(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        return self._module.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        return self._module.ones(shape, dtype=dtype)

    def empty(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        return self._module.empty(shape, dtype=dtype)

    def full(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        return self._module.full(shape, self._module.array(fill_value), dtype=dtype)

    def zeros_like(
        self,
        x: ArrayLikeT | PlaceholderArray | VirtualArray | VirtualArray,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        if isinstance(x, PlaceholderArray) or (
            isinstance(x, VirtualArray) and not x.is_materialized
        ):
            return self.zeros(x.shape, dtype=dtype or x.dtype)
        else:
            return self._module.zeros_like(*materialize_if_virtual(x), dtype=dtype)

    def ones_like(
        self,
        x: ArrayLikeT | PlaceholderArray | VirtualArray,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        if isinstance(x, PlaceholderArray) or (
            isinstance(x, VirtualArray) and not x.is_materialized
        ):
            return self.ones(x.shape, dtype=dtype or x.dtype)
        else:
            return self._module.ones_like(*materialize_if_virtual(x), dtype=dtype)

    def full_like(
        self,
        x: ArrayLikeT | PlaceholderArray | VirtualArray,
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        if isinstance(x, PlaceholderArray) or (
            isinstance(x, VirtualArray) and not x.is_materialized
        ):
            return self.full(x.shape, fill_value, dtype=dtype or x.dtype)
        else:
            return self._module.full_like(
                *materialize_if_virtual(x), self._module.array(fill_value), dtype=dtype
            )

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(start, PlaceholderArray)
        assert not isinstance(stop, PlaceholderArray)
        assert not isinstance(step, PlaceholderArray)
        start, stop, step = materialize_if_virtual(start, stop, step)
        return self._module.arange(start, stop, step, dtype=dtype)

    def meshgrid(
        self, *arrays: ArrayLikeT, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLikeT]:
        return self._module.meshgrid(
            *materialize_if_virtual(*arrays), indexing=indexing
        )

    ############################ testing

    def array_equal(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, equal_nan: bool = False
    ) -> bool:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        if equal_nan:
            # Only newer numpy.array_equal supports the equal_nan parameter.
            both_nan = self._module.logical_and(
                self._module.isnan(x1), self._module.isnan(x2)
            )
            both_equal = x1 == x2
            return self._module.all(self._module.logical_or(both_equal, both_nan))
        else:
            return self._module.array_equal(x1, x2)

    def searchsorted(
        self,
        x: ArrayLikeT,
        values: ArrayLikeT,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(values, PlaceholderArray)
        assert not isinstance(sorter, PlaceholderArray)
        x, values, sorter = materialize_if_virtual(x, values, sorter)
        return self._module.searchsorted(x, values, side=side, sorter=sorter)

    ############################ manipulation
    def apply_ufunc(
        self,
        ufunc: UfuncLike,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> ArrayLikeT | tuple[ArrayLikeT, ...]:
        if method != "__call__" or len(args) == 0:
            raise NotImplementedError

        args = list(materialize_if_virtual(*args))

        if hasattr(ufunc, "resolve_dtypes"):
            return self._apply_ufunc_nep_50(ufunc, method, args, kwargs)
        else:
            return self._apply_ufunc_legacy(ufunc, method, args, kwargs)

    def _get_nep_50_dtype(
        self, obj: Any
    ) -> DType | type[int] | type[complex] | type[float]:
        if hasattr(obj, "dtype"):
            return obj.dtype
        elif isinstance(obj, bool):
            return np.dtype(np.bool_)
        else:
            assert isinstance(obj, (int, complex, float))
            return type(obj)

    # Does NumPy support value-less ufunc resolution?
    def _apply_ufunc_nep_50(
        self,
        ufunc: UfuncLike,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> ArrayLikeT | tuple[ArrayLikeT]:
        # Determine input argument dtypes
        input_arg_dtypes = [self._get_nep_50_dtype(obj) for obj in args]
        # Resolve these for the given ufunc
        arg_dtypes = tuple(input_arg_dtypes + [None] * ufunc.nout)
        resolved_dtypes = ufunc.resolve_dtypes(arg_dtypes)
        # Interpret the arguments under these dtypes, converting scalars to length-1 arrays
        resolved_args = [
            cast("ArrayLikeT", self.asarray(arg, dtype=dtype))
            for arg, dtype in zip(args, resolved_dtypes)
        ]
        # Broadcast to ensure all-scalar or all-nd-array
        broadcasted_args = self.broadcast_arrays(*resolved_args)
        # Allow other nplikes to replace implementation
        impl = self.prepare_ufunc(ufunc)
        # Compute the result
        return impl(*broadcasted_args, **(kwargs or {}))

    # Otherwise, perform default NumPy coercion (value-dependent)
    def _apply_ufunc_legacy(
        self,
        ufunc: UfuncLike,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> ArrayLikeT | tuple[ArrayLikeT]:
        # Convert np.generic to scalar arrays
        resolved_args = [
            cast(
                "ArrayLikeT",
                self.asarray(arg, dtype=arg.dtype if hasattr(arg, "dtype") else None),
            )
            for arg in args
        ]
        broadcasted_args = self.broadcast_arrays(*resolved_args)
        # Choose the broadcasted argument if it wasn't a Python scalar
        non_generic_value_promoted_args = [
            y if hasattr(x, "ndim") else x for x, y in zip(args, broadcasted_args)
        ]
        # Allow other nplikes to replace implementation
        impl = self.prepare_ufunc(ufunc)
        # Compute the result
        return impl(*non_generic_value_promoted_args, **(kwargs or {}))

    def broadcast_arrays(self, *arrays: ArrayLikeT) -> list[ArrayLikeT]:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        arrays = materialize_if_virtual(*arrays)
        return self._module.broadcast_arrays(*arrays)

    def _compute_compatible_shape(
        self, shape: tuple[ShapeItem, ...], existing_shape: tuple[ShapeItem, ...]
    ) -> tuple[ShapeItem, ...]:
        next_shape = list(shape)
        j = None
        length_factor: ShapeItem = 1
        for i, item in enumerate(shape):
            if item != -1:
                length_factor *= item
            elif j is not None:
                raise ValueError("can only have one unknown dimension")
            else:
                j = i
        if j is not None:
            next_shape[j] = math.prod(existing_shape) // length_factor
        return tuple(next_shape)

    def reshape(
        self,
        x: ArrayLikeT | PlaceholderArray | VirtualArray | VirtualArray,
        shape: tuple[ShapeItem, ...],
        *,
        copy: bool | None = None,
    ) -> ArrayLikeT | PlaceholderArray | VirtualArray | VirtualArray:
        if isinstance(x, PlaceholderArray):
            next_shape = self._compute_compatible_shape(shape, x.shape)
            return PlaceholderArray(self, next_shape, x.dtype, x._field_path)
        if isinstance(x, VirtualArray):
            if not x.is_materialized:
                next_shape = self._compute_compatible_shape(shape, x.shape)
                return VirtualArray(
                    self,
                    next_shape,
                    x.dtype,
                    lambda: self.reshape(x.materialize(), next_shape),
                )
            else:
                x = x._array

        if copy is None:
            return self._module.reshape(x, shape)
        elif copy:
            return self._module.reshape(self._module.copy(x, order="C"), shape)
        else:
            result = self._module.asarray(x)
            try:
                result.shape = shape
            except AttributeError:
                raise ValueError("cannot reshape array without copying") from None
            return result

    def shape_item_as_index(self, x1: ShapeItem) -> int:
        if x1 is unknown_length:
            raise TypeError("array module nplikes do not support unknown lengths")
        elif isinstance(x1, int):
            return x1
        else:
            raise TypeError(f"expected None or int type, received {x1}")

    def index_as_shape_item(self, x1: IndexType) -> int:
        return int(x1)

    def derive_slice_for_length(
        self, slice_: slice, length: ShapeItem
    ) -> tuple[IndexType, IndexType, IndexType, ShapeItem]:
        """
        Args:
            slice_: normalized slice object
            length: length of layout

        Return a tuple of (start, stop, step) indices into a layout, suitable for
        `_getitem_range` (if step == 1). Normalize lengths to fit length of array,
        and for arrays with unknown lengths, these offsets become none.
        """
        # We have known_data (therefore known shape), so we can safely convert to int
        slice_as_shape = slice(
            slice_.start
            if slice_.start is None
            else self.index_as_shape_item(slice_.start),
            slice_.stop
            if slice_.stop is None
            else self.index_as_shape_item(slice_.stop),
            slice_.step
            if slice_.step is None
            else self.index_as_shape_item(slice_.step),
        )
        start, stop, step = slice_as_shape.indices(length)
        slice_length = math.ceil((stop - start) / step)

        # Shape items are already valid indices, so we don't need to invoke
        # `shape_item_as_index`
        return start, stop, step, slice_length

    def regularize_index_for_length(
        self, index: IndexType, length: ShapeItem
    ) -> IndexType:
        """
        Args:
            index: index value
            length: length of array

        Returns regularized index that is guaranteed to be in-bounds.
        """  # We have known length and index
        length = cast(int, length)

        if index < 0:
            index = index + length

        if 0 <= index < length:
            return index
        else:
            raise IndexError(f"index value out of bounds (0, {length}): {index}")

    def nonzero(self, x: ArrayLikeT) -> tuple[ArrayLikeT, ...]:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.nonzero(x)

    def where(
        self, condition: ArrayLikeT, x1: ArrayLikeT, x2: ArrayLikeT
    ) -> ArrayLikeT:
        assert not isinstance(condition, PlaceholderArray)
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        condition, x1, x2 = materialize_if_virtual(condition, x1, x2)

        return self._module.where(condition, x1, x2)

    def unique_values(self, x: ArrayLikeT) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        np_unique_accepts_equal_nan = (
            "equal_nan" in inspect.signature(self._module.unique).parameters
        )

        if np_unique_accepts_equal_nan:
            return self._module.unique(
                x,
                return_counts=False,
                return_index=False,
                return_inverse=False,
                equal_nan=False,
            )
        else:
            return self._module.unique(
                x,
                return_counts=False,
                return_index=False,
                return_inverse=False,
            )

    def unique_all(self, x: ArrayLikeT) -> UniqueAllResult:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        values, indices, inverse_indices, counts = self._module.unique(
            x, return_counts=True, return_index=True, return_inverse=True
        )
        # np.unique() flattens inverse indices, but they need to share x's shape
        # See https://github.com/numpy/numpy/issues/20638
        inverse_indices = inverse_indices.reshape(x.shape)
        return UniqueAllResult(values, indices, inverse_indices, counts)

    def sort(
        self,
        x: ArrayLikeT,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        # Note: this keyword argument is different, and the default is different.
        kind = "stable" if stable else "quicksort"
        res = self._module.sort(x, axis=axis, kind=kind)
        if descending:
            return self._module.flip(res, axis=axis)
        else:
            return res

    def concat(
        self,
        arrays: list[ArrayLikeT] | tuple[ArrayLikeT, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLikeT:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        arrays = materialize_if_virtual(*arrays)
        if _nplike_concatenate_has_casting(self._module):
            return self._module.concatenate(arrays, axis=axis, casting="same_kind")
        else:
            return self._module.concatenate(arrays, axis=axis)

    def repeat(
        self,
        x: ArrayLikeT,
        repeats: ArrayLikeT | int,
        *,
        axis: int | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        assert not isinstance(repeats, PlaceholderArray)
        x, repeats = materialize_if_virtual(x, repeats)
        return self._module.repeat(x, repeats=repeats, axis=axis)

    def stack(
        self,
        arrays: list[ArrayLikeT] | tuple[ArrayLikeT, ...],
        *,
        axis: int = 0,
    ) -> ArrayLikeT:
        assert not any(isinstance(x, PlaceholderArray) for x in arrays)
        arrays = materialize_if_virtual(*arrays)
        arrays = list(arrays)
        return self._module.stack(arrays, axis=axis)

    def packbits(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.unpackbits(x, axis=axis, count=count, bitorder=bitorder)

    def broadcast_to(self, x: ArrayLikeT, shape: tuple[ShapeItem, ...]) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.broadcast_to(x, shape)

    def strides(
        self, x: ArrayLikeT | PlaceholderArray | VirtualArray
    ) -> tuple[ShapeItem, ...]:
        if isinstance(x, (PlaceholderArray, VirtualArray)):
            # Assume contiguous
            strides: tuple[ShapeItem, ...] = (x.dtype.itemsize,)
            for item in x.shape[-1:0:-1]:
                strides = (item * strides[0], *strides)
            return strides

        return x.strides  # type: ignore[attr-defined]

    ############################ ufuncs

    def add(
        self, x1: ArrayLikeT, x2: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.add(x1, x2, out=maybe_out)

    def logical_or(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.logical_or(x1, x2, out=maybe_out)

    def logical_and(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.logical_and(x1, x2, out=maybe_out)

    def logical_not(
        self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.logical_not(x, out=maybe_out)

    def sqrt(self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.sqrt(x, out=maybe_out)

    def exp(self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.exp(x, out=maybe_out)

    def divide(
        self, x1: ArrayLikeT, x2: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.divide(x1, x2, out=maybe_out)

    ############################ almost-ufuncs

    def nan_to_num(
        self,
        x: ArrayLikeT,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.nan_to_num(
            x, copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )

    def isclose(
        self,
        x1: ArrayLikeT,
        x2: ArrayLikeT,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayLikeT:
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isnan(self, x: ArrayLikeT) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.isnan(x)

    def all(
        self,
        x: ArrayLikeT,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.all(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def any(
        self,
        x: ArrayLikeT,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.any(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def min(
        self,
        x: ArrayLikeT,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.min(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def max(
        self,
        x: ArrayLikeT,
        *,
        axis: ShapeItem | tuple[ShapeItem, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.max(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def count_nonzero(
        self, x: ArrayLikeT, *, axis: ShapeItem | tuple[ShapeItem, ...] | None = None
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        assert isinstance(axis, int) or axis is None
        (x,) = materialize_if_virtual(x)
        return self._module.count_nonzero(x, axis=axis)

    def cumsum(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.cumsum(x, axis=axis, out=maybe_out)

    def real(self, x: ArrayLikeT) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        xr = self._module.real(x)
        # For numpy, xr is a view on x, but we don't want to mutate x.
        return self._module.copy(xr)

    def imag(self, x: ArrayLikeT) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        xr = self._module.imag(x)
        # For numpy, xr is a view on x, but we don't want to mutate x.
        return self._module.copy(xr)

    def angle(self, x: ArrayLikeT, deg: bool = False) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.angle(x, deg)

    def round(self, x: ArrayLikeT, decimals: int = 0) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.round(x, decimals=decimals)

    def array_str(
        self,
        x: ArrayLikeT,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        if isinstance(x, PlaceholderArray) or (
            isinstance(x, VirtualArray) and not x.is_materialized
        ):
            return "[## ... ##]"
        (x,) = materialize_if_virtual(x)
        return self._module.array_str(
            x,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def astype(
        self, x: ArrayLikeT, dtype: DTypeLike, *, copy: bool | None = True
    ) -> ArrayLikeT:
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return x.astype(dtype, copy=copy)  # type: ignore[attr-defined]

    def can_cast(
        self, from_: DTypeLike | ArrayLikeT, to: DTypeLike | ArrayLikeT
    ) -> bool:
        return self._module.can_cast(from_, to, casting="same_kind")

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return cls.is_own_array_type(type(obj))
