# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import numpy

from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.shape import ShapeItem
from awkward._singleton import PublicSingleton
from awkward._typing import (
    TYPE_CHECKING,
    Any,
    DType,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
)
from awkward.errors import AxisError

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from awkward._nplikes.placeholder import PlaceholderArray


IndexType: TypeAlias = "int | ArrayLikeT"


class UniqueAllResult(NamedTuple):
    values: ArrayLike
    indices: ArrayLike
    inverse_indices: ArrayLike
    counts: ArrayLike


class NumpyMetadata(PublicSingleton):
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    longlong = numpy.longlong
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    str_ = numpy.str_
    bytes_ = numpy.bytes_

    datetime64 = numpy.datetime64
    timedelta64 = numpy.timedelta64

    intp = numpy.intp
    integer = numpy.integer
    signedinteger = numpy.signedinteger
    unsignedinteger = numpy.unsignedinteger
    floating = numpy.floating
    complexfloating = numpy.complexfloating
    number = numpy.number
    object_ = numpy.object_
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    finfo = numpy.finfo
    errstate = numpy.errstate
    newaxis: None = numpy.newaxis

    ndarray = numpy.ndarray

    nan = numpy.nan
    inf = numpy.inf

    nat = numpy.datetime64("NaT")
    datetime_data = staticmethod(numpy.datetime_data)
    issubdtype = staticmethod(numpy.issubdtype)

    AxisError = AxisError


if hasattr(numpy, "float16"):
    NumpyMetadata.float16 = numpy.float16  # type: ignore[attr-defined]

if hasattr(numpy, "float128"):
    NumpyMetadata.float128 = numpy.float128  # type: ignore[attr-defined]

if hasattr(numpy, "complex256"):
    NumpyMetadata.complex256 = numpy.complex256  # type: ignore[attr-defined]


ArrayLikeT = TypeVar("ArrayLikeT", bound=ArrayLike)


class UfuncLike(Protocol):
    nargs: int
    nin: int
    nout: int

    def resolve_dtypes(
        self, dtypes: tuple[DType | type[int] | type[complex] | type[float] | None, ...]
    ) -> tuple[DType, ...]: ...

    def __call__(self, *args: ArrayLikeT, **kwargs) -> ArrayLikeT: ...


class NumpyLike(PublicSingleton, Protocol[ArrayLikeT]):
    ############################ Awkward features

    @abstractmethod
    def apply_ufunc(
        self,
        ufunc: UfuncLike,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> ArrayLikeT | tuple[ArrayLikeT, ...]: ...

    @property
    @abstractmethod
    def supports_structured_dtypes(self) -> bool: ...

    @property
    @abstractmethod
    def known_data(self) -> bool: ...

    @property
    @abstractmethod
    def is_eager(self) -> bool: ...

    ############################ array creation

    @property
    @abstractmethod
    def ndarray(self) -> ArrayLikeT: ...

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> ArrayLikeT | PlaceholderArray: ...

    # FIXME: find a way to express TypeVar(..., OtherTypeVar(...), FOO) such that
    #        this function preserves the type identity of the input
    @abstractmethod
    def ascontiguousarray(
        self, x: ArrayLikeT | PlaceholderArray
    ) -> ArrayLikeT | PlaceholderArray: ...

    @abstractmethod
    def frombuffer(
        self, buffer, *, dtype: DTypeLike | None = None, count: ShapeItem = -1
    ) -> ArrayLikeT: ...

    @abstractmethod
    def from_dlpack(self, x: Any) -> ArrayLikeT: ...

    @abstractmethod
    def zeros(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def ones(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def empty(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def full(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def zeros_like(
        self, x: ArrayLikeT | PlaceholderArray, *, dtype: DTypeLike | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def ones_like(
        self, x: ArrayLikeT | PlaceholderArray, *, dtype: DTypeLike | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def full_like(
        self,
        x: ArrayLikeT | PlaceholderArray,
        fill_value,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: DTypeLike | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def meshgrid(
        self, *arrays: ArrayLikeT, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLikeT]: ...

    ############################ testing

    @abstractmethod
    def array_equal(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, equal_nan: bool = False
    ) -> bool: ...

    @abstractmethod
    def searchsorted(
        self,
        x: ArrayLikeT,
        values: ArrayLikeT,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    ############################ manipulation

    @abstractmethod
    def broadcast_arrays(self, *arrays: ArrayLikeT) -> list[ArrayLikeT]: ...

    @abstractmethod
    def broadcast_to(
        self, x: ArrayLikeT, shape: tuple[ShapeItem, ...]
    ) -> ArrayLikeT: ...

    @abstractmethod
    def shape_item_as_index(self, x1: ShapeItem) -> int | ArrayLikeT: ...

    @abstractmethod
    def index_as_shape_item(self, x1: IndexType) -> ShapeItem: ...

    @abstractmethod
    def derive_slice_for_length(
        self, slice_: slice, length: ShapeItem
    ) -> tuple[IndexType, IndexType, IndexType, ShapeItem]: ...

    @abstractmethod
    def regularize_index_for_length(
        self, index: IndexType, length: ShapeItem
    ) -> IndexType: ...

    # FIXME: find a way to express TypeVar(..., OtherTypeVar(...), FOO) such that
    #        this function preserves the type identity of the input
    @abstractmethod
    def reshape(
        self,
        x: ArrayLikeT | PlaceholderArray,
        shape: tuple[ShapeItem, ...],
        *,
        copy: bool | None = None,
    ) -> ArrayLikeT | PlaceholderArray: ...

    @abstractmethod
    def nonzero(self, x: ArrayLikeT) -> tuple[ArrayLikeT, ...]: ...

    @abstractmethod
    def where(
        self, condition: ArrayLikeT, x1: ArrayLikeT, x2: ArrayLikeT
    ) -> ArrayLikeT: ...

    @abstractmethod
    def unique_values(self, x: ArrayLikeT) -> ArrayLikeT: ...

    @abstractmethod
    def unique_all(self, x: ArrayLikeT) -> UniqueAllResult: ...

    @abstractmethod
    def sort(
        self,
        x: ArrayLikeT,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def concat(
        self,
        arrays: list[ArrayLikeT] | tuple[ArrayLikeT, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def repeat(
        self,
        x: ArrayLikeT,
        repeats: ArrayLikeT | int,
        *,
        axis: int | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def stack(
        self,
        arrays: list[ArrayLikeT] | tuple[ArrayLikeT, ...],
        *,
        axis: int = 0,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def packbits(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLikeT: ...

    @abstractmethod
    def unpackbits(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLikeT: ...

    @abstractmethod
    def strides(self, x: ArrayLikeT | PlaceholderArray) -> tuple[ShapeItem, ...]: ...

    ############################ ufuncs

    @abstractmethod
    def add(
        self, x1: ArrayLikeT, x2: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def logical_or(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def logical_and(
        self, x1: ArrayLikeT, x2: ArrayLikeT, *, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def logical_not(
        self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def sqrt(
        self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def exp(self, x: ArrayLikeT, maybe_out: ArrayLikeT | None = None) -> ArrayLikeT: ...

    @abstractmethod
    def divide(
        self, x1: ArrayLikeT, x2: ArrayLikeT, maybe_out: ArrayLikeT | None = None
    ) -> ArrayLikeT: ...

    ############################ almost-ufuncs

    @abstractmethod
    def nan_to_num(
        self,
        x: ArrayLikeT,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def isclose(
        self,
        x1: ArrayLikeT,
        x2: ArrayLikeT,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def isnan(self, x: ArrayLikeT) -> ArrayLikeT: ...

    @abstractmethod
    def all(
        self,
        x: ArrayLikeT,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def any(
        self,
        x: ArrayLikeT,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def min(
        self,
        x: ArrayLikeT,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def max(
        self,
        x: ArrayLikeT,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def count_nonzero(
        self, x: ArrayLikeT, *, axis: int | tuple[int, ...] | None = None
    ) -> ArrayLikeT: ...

    @abstractmethod
    def cumsum(
        self,
        x: ArrayLikeT,
        *,
        axis: int | None = None,
        maybe_out: ArrayLikeT | None = None,
    ) -> ArrayLikeT: ...

    @abstractmethod
    def array_str(
        self,
        x: ArrayLikeT,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ): ...

    @abstractmethod
    def astype(
        self, x: ArrayLikeT, dtype: DTypeLike, *, copy: bool | None = True
    ) -> ArrayLikeT: ...

    @abstractmethod
    def can_cast(self, from_: DType | ArrayLikeT, to: DType | ArrayLikeT) -> bool: ...

    @abstractmethod
    def is_c_contiguous(self, x: ArrayLikeT | PlaceholderArray) -> bool: ...

    @abstractmethod
    def real(self, x: ArrayLikeT) -> ArrayLikeT: ...

    @abstractmethod
    def imag(self, x: ArrayLikeT) -> ArrayLikeT: ...

    @abstractmethod
    def angle(self, x: ArrayLikeT, deg: bool = False) -> ArrayLikeT: ...

    @abstractmethod
    def round(self, x: ArrayLikeT, decimals: int = 0) -> ArrayLikeT: ...

    @classmethod
    @abstractmethod
    def is_own_array(cls, obj) -> bool: ...

    @classmethod
    @abstractmethod
    def is_own_array_type(cls, type_: type) -> bool: ...
