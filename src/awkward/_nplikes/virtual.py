# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import typing as tp
from functools import reduce
from operator import mul

from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._typing import TYPE_CHECKING, Any, DType, Self
from awkward._util import Sentinel

np = NumpyMetadata.instance()

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


_unmaterialized = Sentinel("<unmaterialized>", None)


def materialize_if_virtual(*arrays: VirtualArray | ArrayLike) -> tuple[ArrayLike, ...]:
    """
    A little helper function to materialize all virtual arrays in a list of arrays.
    This is useful to materialize all potential virtual arrays right before calling
    an awkward_cpp kernel, e.g. `src/awkward/contents/numpyarray.py (def _argsort_next())`
    can be modified to:

        .. code-block:: python

            ...
            self._backend[
                "awkward_sorting_ranges_length",
                _offsets_length.dtype.type,
                parents.dtype.type,
            ](
                *materialize_if_virtual(
                    _offsets_length.data,
                    parents.data,
                ),
                parents_length,
            )
            ...
    """
    return tuple(
        array.materialize() if isinstance(array, VirtualArray) else array
        for array in arrays
    )


class VirtualArray(ArrayLike):
    # let's keep track of the form keys that have been materialized.
    #
    # In future, we could track even more, like the number of times
    # a form key has been materialized, etc.
    #
    # (TODO: Is this set supposed to be thread-local?)
    _materialized_form_keys: tp.ClassVar[set] = set()

    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DType,
        generator: tp.Callable[
            [], ArrayLike
        ],  # annotation (should) make clear that it's a callable without(!) arguments that returns an ArrayLike
        form_key: str | None = None,
    ) -> None:
        # array metadata
        self._nplike = nplike
        self._shape = shape
        self._dtype = np.dtype(dtype)

        # this is the _actual_ array, if it's
        # not materialized, it's a sentinel
        self._array = _unmaterialized

        # the generator function that creates the array
        self._generator = generator

        # the form key of the array in the ak.forms.Form
        self._form_key = form_key

    @property
    def nplike(self) -> NumpyLike:
        return self._nplike

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._shape

    @property
    def size(self) -> ShapeItem:
        return reduce(mul, self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def nbytes(self) -> int:
        if self.is_materialized:
            return self._array.nbytes
        return 0

    @property
    def T(self):
        transposed = type(self)(
            nplike=self.nplike,
            shape=self.shape[::-1],
            dtype=self.dtype,
            # we can delay materialization by stacking the transposition with a lambda
            generator=lambda: self.generator().T,
            form_key=self.form_key,
        )
        if self.is_materialized:
            transposed._array = self._array.T
        return transposed

    @property
    def generator(self) -> tp.Callable:
        return self._generator

    @property
    def form_key(self) -> str | None:
        return self._form_key

    def materialize(self) -> ArrayLike:
        if self._array is _unmaterialized:
            print("Materializing:", self.form_key)  # debugging purposes
            self._materialized_form_keys.add(self.form_key)
            self._array = self.nplike.asarray(self.generator())
        return tp.cast(ArrayLike, self._array)

    @property
    def is_materialized(self) -> bool:
        return self._array is not _unmaterialized

    def __repr__(self):
        return f"{self.__class__.__name__}(array={self._array}, shape={self.shape} dtype={self.dtype})"

    def __setitem__(self, key, value):
        array = self.materialize()
        array[key] = value

    def __getitem__(self, key):
        array = self.materialize()
        return array[key]

    # TODO: The following methods need to be implemented (ArrayLike protocol).
    # If there's something missing, we should take a look at the
    # typetracer implementation in src/awkward/_nplikes/typetracer.py
    #
    # Note: not all of the following methods need materialization, e.g. __len__.
    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __int__(self) -> int:
        raise NotImplementedError

    def __index__(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def view(self, dtype: DTypeLike) -> Self:
        raise NotImplementedError

    def __add__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __sub__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __mul__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __truediv__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __floordiv__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __gt__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __lt__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __ge__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __le__(self, other: int | complex | float | Self) -> Self:
        raise NotImplementedError

    def __eq__(self, other: int | complex | float | bool | Self) -> Self:
        raise NotImplementedError

    def __and__(self, other: int | bool | Self) -> Self:
        raise NotImplementedError

    def __or__(self, other: int | bool | Self) -> Self:
        raise NotImplementedError

    def __invert__(self) -> Self:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __dlpack_device__(self) -> tuple[int, int]:
        raise NotImplementedError

    def __dlpack__(self, stream: Any = None) -> Any:
        raise NotImplementedError
