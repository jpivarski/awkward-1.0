# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import reduce
from operator import mul

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._operators import NDArrayOperatorsMixin
from awkward._typing import TYPE_CHECKING, Any, Callable, ClassVar, DType, Self, cast
from awkward._util import Sentinel

np = NumpyMetadata.instance()

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


UNMATERIALIZED = Sentinel("<UNMATERIALIZED>", None)


def materialize_if_virtual(*args: Any) -> tuple[Any, ...]:
    """
    A little helper function to materialize all virtual arrays in a list of arrays.
    """
    return tuple(
        arg.materialize() if isinstance(arg, VirtualArray) else arg for arg in args
    )


class VirtualArray(NDArrayOperatorsMixin, ArrayLike):
    # let's keep track of the form keys that have been materialized.
    #
    # In future, we could track even more, like the number of times
    # a form key has been materialized, etc.
    #
    # (TODO: Is this set supposed to be thread-local?)
    _materialized_form_keys: ClassVar[set] = set()

    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DType,
        generator: Callable[[], ArrayLike],
        form_key: str | None = None,
    ) -> None:
        if not isinstance(nplike, (ak._nplikes.numpy.Numpy, ak._nplikes.cupy.Cupy)):
            raise ValueError(
                f"Only numpy and cupy nplikes are supported for VirtualArray. Received {type(nplike)}"
            )

        # array metadata
        self._nplike = nplike
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._array: Sentinel | ArrayLike = UNMATERIALIZED
        self._generator = generator
        self._form_key = form_key

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> ShapeItem:
        return reduce(mul, self._shape)

    @property
    def nbytes(self) -> ShapeItem:
        if self.is_materialized:
            return cast(ArrayLike, self._array).nbytes
        return 0

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        out: tuple[ShapeItem, ...] = (self._dtype.itemsize,)
        for item in reversed(self._shape):
            out = (item * out[0], *out)
        return out

    def materialize(self) -> ArrayLike:
        if self._array is UNMATERIALIZED:
            self._materialized_form_keys.add(self.form_key)
            self._array = cast(ArrayLike, self._nplike.asarray(self.generator()))
        return cast(ArrayLike, self._array)

    @property
    def is_materialized(self) -> bool:
        return self._array is not UNMATERIALIZED

    @property
    def T(self):
        if self.is_materialized:
            return self._array.T

        return type(self)(
            self._nplike,
            self._shape[::-1],
            self._dtype,
            lambda: self.materialize().T,
            self._form_key,
        )

    def view(self, dtype: DTypeLike) -> Self:
        # TODO: Should views return a view of the underlying NDArray if it's materialized?
        dtype = np.dtype(dtype)
        if len(self._shape) >= 1:
            last, remainder = divmod(
                self._shape[-1] * self._dtype.itemsize, dtype.itemsize
            )
            if remainder is not unknown_length and remainder != 0:
                raise ValueError(
                    "new size of array with larger dtype must be a "
                    "divisor of the total size in bytes (of the last axis of the array)"
                )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        return type(self)(
            self._nplike,
            shape,
            dtype,
            lambda: self.materialize().view(dtype),
            self._form_key,
        )

    @property
    def generator(self) -> Callable:
        return self._generator

    @property
    def form_key(self) -> str | None:
        return self._form_key

    @form_key.setter
    def form_key(self, value: str | None):
        if value is not None and not isinstance(value, str):
            raise TypeError("form_key must be None or a string")
        self._form_key = value

    @property
    def nplike(self) -> NumpyLike:
        return self._nplike

    def copy(self) -> VirtualArray:
        self.materialize()
        return self

    def tolist(self) -> NumpyLike:
        return self.materialize().tolist()

    @property
    def ctypes(self):
        if isinstance((self._nplike), ak._nplikes.cupy.Cupy):
            raise AttributeError("Cupy ndarrays do not have a ctypes attribute.")
        return self.materialize().ctypes

    @property
    def data(self):
        return self.materialize().data

    def __array__(self, *args, **kwargs):
        raise AssertionError(
            "The '__array__' method should never be called directly on a VirtualArray."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.nplike.apply_ufunc(ufunc, method, inputs, kwargs)

    def __repr__(self):
        dtype = repr(self._dtype)
        if self.shape is None:
            shape = ""
        else:
            shape = ", shape=" + repr(self._shape)
        return f"VirtualArray(array={self._array}, {dtype}{shape})"

    def __str__(self):
        if self.ndim == 0:
            return "##"
        else:
            return repr(self)

    def __getitem__(self, index):
        if self.is_materialized:
            return self._array.__getitem__(index)

        if isinstance(index, slice):
            length = self._shape[0]

            if length is unknown_length:
                return self.materialize().__getitem__(index)
            elif (
                index.start is unknown_length
                or index.stop is unknown_length
                or index.step is unknown_length
            ):
                return self.materialize().__getitem__(index)
            else:
                start, stop, step = index.indices(length)
                new_length = (stop - start) // step

            return type(self)(
                self._nplike,
                (new_length,),
                self._dtype,
                lambda: self.materialize()[index],
                self._form_key,
            )
        else:
            return self.materialize().__getitem__(index)

    def __setitem__(self, key, value):
        array = self.materialize()
        array.__setitem__(key, value)

    def __bool__(self) -> bool:
        array = self.materialize()
        return bool(array)

    def __int__(self) -> int:
        array = self.materialize()
        if array.ndim == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be converted to an int.")

    def __index__(self) -> int:
        array = self.materialize()
        if array.ndim == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be used as an index.")

    def __len__(self) -> int:
        return int(self._shape[0])

    def __iter__(self):
        array = self.materialize()
        return iter(array)

    def __dlpack_device__(self) -> tuple[int, int]:
        array = self.materialize()
        return array.__dlpack_device__()

    def __dlpack__(self, *args, **kwargs):
        array = self.materialize()
        if args or kwargs:
            return array.__dlpack__(*args, **kwargs)
        return array.__dlpack__()
