# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import reduce
from operator import mul

from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._typing import TYPE_CHECKING, Any, DType, Self

np = NumpyMetadata.instance()

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class PlaceholderArray(ArrayLike):
    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DType,
        field_path: tuple[str, ...] = (),
    ):
        self._nplike = nplike
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._field_path = field_path

    @property
    def field_path(self) -> str:
        return ".".join(self._field_path)

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
    def nbytes(self) -> int:
        return 0

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        out: tuple[ShapeItem, ...] = (self._dtype.itemsize,)
        for item in reversed(self._shape):
            out = (item * out[0], *out)
        return out

    @property
    def T(self):
        return type(self)(self._nplike, self._shape[::-1], self._dtype)

    def view(self, dtype: DTypeLike) -> Self:
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
        return type(self)(self._nplike, shape, dtype, self._field_path)

    def __repr__(self):
        dtype = repr(self._dtype)
        if self.shape is None:
            shape = ""
        else:
            shape = ", shape=" + repr(self._shape)
        return f"PlaceholderArray({dtype}{shape})"

    def __getitem__(self, index):
        # Typetracers permit slices that don't touch data or shapes
        if isinstance(index, slice):
            length = self._shape[0]

            # Unknown-length placeholders should not be sliced (as their shapes would be touched(
            if length is unknown_length:
                raise AssertionError(
                    "placeholder arrays that are sliced should have known shapes"
                )
            # Known-length placeholders *always* need a known shape
            elif (
                index.start is unknown_length
                or index.stop is unknown_length
                or index.step is unknown_length
            ):
                raise AssertionError(
                    "known-length placeholders should never encounter unknown lengths in slices"
                )
            else:
                start, stop, step = index.indices(length)
                new_length = (stop - start) // step

            return type(self)(
                self._nplike, (new_length,), self._dtype, self._field_path
            )
        else:
            msg = f"{type(self).__name__} supports only trivial slices, not {type(index).__name__}"
            if self.field_path:
                msg += f"\n\nAwkward-array attempted to access a field '{self.field_path}', but "
                msg += (
                    "it has been excluded during a pre-run phase (possibly by Dask). "
                )
                msg += "If this was supposed to happen automatically (e.g. you're using Dask), "
                msg += "please report it to the developers at: https://github.com/scikit-hep/awkward/issues"
            raise TypeError(msg)

    def __setitem__(self, key, value):
        raise RuntimeError

    def __bool__(self) -> bool:
        raise RuntimeError

    def __int__(self) -> int:
        raise RuntimeError

    def __index__(self) -> int:
        raise RuntimeError

    def __len__(self) -> int:
        return int(self._shape[0])

    def __add__(self, other):
        raise RuntimeError

    def __and__(self, other):
        raise RuntimeError

    def __eq__(self, other):
        raise RuntimeError

    def __floordiv__(self, other):
        raise RuntimeError

    def __ge__(self, other):
        raise RuntimeError

    def __gt__(self, other):
        raise RuntimeError

    def __invert__(self):
        raise RuntimeError

    def __le__(self, other):
        raise RuntimeError

    def __lt__(self, other):
        raise RuntimeError

    def __mul__(self, other):
        raise RuntimeError

    def __or__(self, other):
        raise RuntimeError

    def __sub__(self, other):
        raise RuntimeError

    def __truediv__(self, other):
        raise RuntimeError

    __iter__: None = None

    def __dlpack_device__(self) -> tuple[int, int]:
        raise RuntimeError

    def __dlpack__(self, stream: Any = None) -> Any:
        raise RuntimeError
