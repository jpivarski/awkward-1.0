# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of_obj
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._layout import maybe_posaxis
from awkward._meta.numpymeta import NumpyMeta
from awkward._nplikes import to_nplike
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import TypeTracerArray
from awkward._nplikes.virtual import VirtualArray
from awkward._parameters import (
    parameters_intersect,
    type_parameters_equal,
)
from awkward._regularize import is_integer_like
from awkward._slicing import NO_HEAD
from awkward._typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Self,
    SupportsIndex,
    final,
)
from awkward._util import UNSET
from awkward.contents.content import (
    ApplyActionOptions,
    Content,
    ImplementsApplyAction,
    RemoveStructureOptions,
    ToArrowOptions,
)
from awkward.errors import AxisError
from awkward.forms.form import Form, FormKeyPathT
from awkward.forms.numpyform import NumpyForm
from awkward.index import Index
from awkward.types.numpytype import primitive_to_dtype

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class NumpyArray(NumpyMeta, Content):
    """
    A NumpyArray describes 1-dimensional or rectilinear data using a NumPy
    `np.ndarray`, a CuPy `cp.ndarray`, etc., depending on the backend.

    This class is aware of the rectilinear array's `shape` and `strides`, and
    allows for arbitrary `strides`, such as Fortran-ordered data. However, many
    operations require C-contiguous data, so derivatives of Fortran-ordered
    arrays may not be Fortran-ordered.

    Only a subset of `dtype` values are allowed, and only for your system's
    native endianness:

    * `bool`: boolean, like NumPy's `np.bool_` (considered distinct from integers)
    * `int8`: signed 8-bit
    * `uint8`: unsigned 8-bit
    * `int16`: signed 16-bit
    * `uint16`: unsigned 16-bit
    * `int32`: signed 32-bit
    * `uint32`: unsigned 32-bit
    * `int64`: signed 64-bit
    * `uint64`: unsigned 64-bit
    * `float16`: floating point 16-bit, if your system's NumPy supports it
    * `float32`: floating point 32-bit
    * `float64`: floating point 64-bit
    * `float128`: floating point 128-bit, if your system's NumPy supports it
    * `complex64`: floating complex numbers composed of 32-bit real/imag parts
    * `complex128`: floating complex numbers composed of 64-bit real/imag parts
    * `complex256`: floating complex numbers composed of 128-bit real/imag parts, if your system's NumPy supports it
    * `datetime64`: date/time, origin is midnight on January 1, 1970, in any units NumPy supports
    * `timedelta64`: time difference, in any units NumPy supports

    If the `shape` is one-dimensional, a NumpyArray corresponds to an Apache
    Arrow [Primitive array](https://arrow.apache.org/docs/format/Columnar.html#fixed-size-primitive-layout).

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class NumpyArray(Content):
            def __init__(self, data):
                assert isinstance(data, numpy_like_array)
                assert data.dtype in allowed_dtypes
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, where):
                result = self.data[where]
                if isinstance(result, numpy_like_array):
                    return NumpyArray(result)
                else:
                    return result
    """

    def __init__(self, data: ArrayLike, *, parameters=None, backend=None):
        if backend is None:
            backend = backend_of_obj(data, default=NumpyBackend.instance())

        self._data = backend.nplike.asarray(data)

        if not isinstance(backend.nplike, Jax):
            ak.types.numpytype.dtype_to_primitive(self._data.dtype)

        if len(self._data.shape) == 0:
            raise TypeError(
                f"{type(self).__name__} 'data' must be an array, not a scalar: {data!r}"
            )

        if parameters is not None and parameters.get("__array__") in ("char", "byte"):
            if data.dtype != np.dtype(np.uint8) or len(data.shape) != 1:
                raise ValueError(
                    "{} is a {}, so its 'data' must be 1-dimensional and uint8, not {}".format(
                        type(self).__name__, parameters["__array__"], repr(data)
                    )
                )

        self._init(parameters, backend)

    @property
    def data(self) -> ArrayLike:
        return self._data

    form_cls: Final = NumpyForm

    def copy(
        self,
        data=UNSET,
        *,
        parameters=UNSET,
        backend=UNSET,
    ):
        return NumpyArray(
            self._data if data is UNSET else data,
            parameters=self._parameters if parameters is UNSET else parameters,
            backend=self._backend if backend is UNSET else backend,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            data=copy.deepcopy(self._data, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, data, *, parameters=None, backend=None):
        return cls(data, parameters=parameters, backend=backend)

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._data.shape

    @property
    def inner_shape(self) -> tuple[ShapeItem, ...]:
        if hasattr(self._data, "inner_shape"):
            inner_shape = self._data.inner_shape
        else:
            inner_shape = self._data.shape[1:]
        return inner_shape

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        return self._backend.nplike.strides(self._data)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def _raw(self, nplike=None):
        return to_nplike(self.data, nplike, from_nplike=self._backend.nplike)

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> NumpyForm:
        return self.form_cls(
            ak.types.numpytype.dtype_to_primitive(self._data.dtype),
            self.inner_shape,
            parameters=self._parameters,
            form_key=getkey(self),
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> NumpyForm:
        return self.form_cls(
            ak.types.numpytype.dtype_to_primitive(self._data.dtype),
            self.inner_shape,
            parameters=self._parameters,
            form_key=repr(path),
        )

    def _to_buffers(
        self,
        form: Form,
        getkey: Callable[[Content, Form, str], str],
        container: MutableMapping[str, ArrayLike],
        backend: Backend,
        byteorder: str,
    ):
        assert isinstance(form, self.form_cls)
        key = getkey(self, form, "data")
        container[key] = ak._util.native_to_byteorder(
            self._raw(backend.nplike), byteorder
        )

    def _to_typetracer(self, forget_length: bool) -> Self:
        backend = TypeTracerBackend.instance()
        data = self._raw(backend.nplike)
        return NumpyArray(
            data.forget_length() if forget_length else data,
            parameters=self._parameters,
            backend=backend,
        )

    def _touch_data(self, recursive: bool):
        if not self._backend.nplike.known_data:
            self._data.touch_data()

    def _touch_shape(self, recursive: bool):
        if not self._backend.nplike.known_data:
            self._data.touch_shape()

    @property
    def length(self) -> ShapeItem:
        return self._data.shape[0]

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=" + repr(str(self._data.shape[0])))
        else:
            out.append(
                " shape='({})'".format(", ".join(str(x) for x in self._data.shape))
            )

        extra = self._repr_extra(indent + "    ")

        arraystr_lines = self._backend.nplike.array_str(
            self._data, max_line_width=30
        ).split("\n")

        if len(extra) != 0 or len(arraystr_lines) > 1:
            arraystr_lines = self._backend.nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.extend(extra)
            out.append("\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def to_RegularArray(self):
        shape = self._data.shape
        zeroslen = [1]
        for x in shape:
            zeroslen.append(zeroslen[-1] * x)

        out = NumpyArray(
            self._backend.nplike.reshape(self._data, (-1,)),
            parameters=None,
            backend=self._backend,
        )
        for i in range(len(shape) - 1, 0, -1):
            out = ak.contents.RegularArray(out, shape[i], zeroslen[i], parameters=None)
        out._parameters = self._parameters
        return out

    def maybe_to_NumpyArray(self) -> Self:
        return self

    def __iter__(self):
        return iter(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            self._backend.nplike.reshape(tmp, ((0,) + tmp.shape[2:])),
            parameters=None,
            backend=self._backend,
        )

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder = isinstance(self._data, PlaceholderArray)
        return is_placeholder

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual = (
            isinstance(self._data, VirtualArray) and not self._data.is_materialized
        )
        return is_virtual

    def _getitem_at(self, where: IndexType):
        if not self._backend.nplike.known_data and len(self._data.shape) == 1:
            self._touch_data(recursive=False)
            return TypeTracerArray._new(self._data.dtype, shape=())

        try:
            out = self._data[where]
        except IndexError as err:
            raise ak._errors.index_error(self, where, str(err)) from err

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, parameters=None, backend=self._backend)
        else:
            return out

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        try:
            out = self._data[start:stop]
        except IndexError as err:
            raise ak._errors.index_error(self, slice(start, stop), str(err)) from err

        return NumpyArray(out, parameters=self._parameters, backend=self._backend)

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        raise ak._errors.index_error(self, where, "not an array of records")

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        if len(where) == 0:
            return self._getitem_range(0, 0)
        raise ak._errors.index_error(self, where, "not an array of records")

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)
        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err
        return NumpyArray(nextdata, parameters=self._parameters, backend=self._backend)

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        if self._data.ndim == 1:
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.to_RegularArray()
            return next._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            )

    def _getitem_next(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        if head is NO_HEAD:
            return self

        elif is_integer_like(head):
            where = (slice(None), head, *tail)

            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head, *tail), str(err)) from err

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, parameters=None, backend=self._backend)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head, *tail)
            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head, *tail), str(err)) from err

            return NumpyArray(out, parameters=self._parameters, backend=self._backend)

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak.index.Index64):
            if advanced is None:
                where = (slice(None), head.data, *tail)
            else:
                where = (
                    self._backend.index_nplike.asarray(advanced.data),
                    head.data,
                    *tail,
                )

            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head, *tail), str(err)) from err

            return NumpyArray(out, parameters=self._parameters, backend=self._backend)

        elif isinstance(head, ak.contents.ListOffsetArray):
            where = (slice(None), head, *tail)
            try:
                out = self._data[where]
            except IndexError as err:
                raise ak._errors.index_error(self, (head, *tail), str(err)) from err

            return NumpyArray(out, parameters=self._parameters, backend=self._backend)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            next = self.to_RegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError("axis=0 not allowed for flatten")

        elif len(self.shape) != 1:
            return self.to_RegularArray()._offsets_and_flattened(axis, depth)

        else:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

    def _mergeable_next(self, other: Content, mergebool: bool) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # Is the other array indexed or optional?
        elif other.is_indexed or other.is_option:
            return self._mergeable_next(other.content, mergebool)
        # Otherwise, do the parameters match? If not, we can't merge.
        elif not type_parameters_equal(self._parameters, other._parameters):
            return False
        # Simplify *this* branch to be 1D self
        elif len(self.shape) > 1:
            return self._to_regular_primitive()._mergeable_next(other, mergebool)

        elif isinstance(other, ak.contents.NumpyArray):
            if self._data.ndim != other._data.ndim:
                return False

            # Obvious fast-path
            if self.dtype == other.dtype:
                return True

            # Special-case booleans i.e. {bool, number}
            elif (
                np.issubdtype(self.dtype, np.bool_)
                and np.issubdtype(other.dtype, np.number)
            ) or (
                np.issubdtype(self.dtype, np.number)
                and np.issubdtype(other.dtype, np.bool_)
            ):
                return mergebool

            # Currently we're less permissive than NumPy on merging datetimes / timedeltas
            elif (
                np.issubdtype(self.dtype, np.datetime64)
                or np.issubdtype(self.dtype, np.timedelta64)
                or np.issubdtype(other.dtype, np.datetime64)
                or np.issubdtype(other.dtype, np.timedelta64)
            ):
                return False

            # Default merging (can we cast one to the other)
            else:
                return self.backend.nplike.can_cast(
                    self.dtype, other.dtype
                ) or self.backend.nplike.can_cast(other.dtype, self.dtype)

        else:
            return False

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self

        if len(self.shape) > 1:
            return self.to_RegularArray()._mergemany(others)

        head, tail = self._merging_strategy(others)

        contiguous_arrays = []

        parameters = self._parameters
        for array in head:
            if isinstance(array, ak.contents.EmptyArray):
                continue

            parameters = parameters_intersect(parameters, array._parameters)
            if isinstance(array, ak.contents.NumpyArray):
                contiguous_arrays.append(array.data)
            else:
                raise AssertionError(
                    "cannot merge "
                    + type(self).__name__
                    + " with "
                    + type(array).__name__
                )

        contiguous_arrays = self._backend.nplike.concat(contiguous_arrays)

        next = NumpyArray(
            contiguous_arrays, parameters=parameters, backend=self._backend
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        return self

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif len(self.shape) <= 1:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        else:
            return self.to_RegularArray()._local_index(axis, depth)

    def to_contiguous(self) -> Self:
        if self.is_contiguous:
            return self
        else:
            return ak.contents.NumpyArray(
                self._backend.nplike.ascontiguousarray(self._data),
                parameters=self._parameters,
                backend=self._backend,
            )

    @property
    def is_contiguous(self) -> bool:
        return self._backend.nplike.is_c_contiguous(self._data)

    def _subranges_equal(self, starts, stops, length, sorted=True):
        is_equal = ak.index.Index64.zeros(1, nplike=self._backend.nplike)

        assert (
            starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
        )
        if self.dtype == np.bool_:
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_NumpyArray_subrange_equal_bool",
                    self.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                    np.bool_,
                ](
                    self._backend.nplike.astype(
                        self._data, dtype=self.dtype, copy=True
                    ),
                    starts.data,
                    stops.data,
                    starts.length,
                    is_equal.data,
                )
            )
        else:
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_NumpyArray_subrange_equal",
                    self.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                    np.bool_,
                ](
                    self._backend.nplike.astype(
                        self._data, dtype=self.dtype, copy=True
                    ),
                    starts.data,
                    stops.data,
                    starts.length,
                    is_equal.data,
                )
            )

        return True if is_equal[0] == 1 else False

    def _as_unique_strings(self, offsets):
        offsets = ak.index.Index64(offsets.data, nplike=offsets.nplike)
        outoffsets = ak.index.Index64.empty(
            offsets.length, nplike=self._backend.index_nplike
        )
        out = self._backend.nplike.empty(self.shape[0], dtype=self.dtype)

        assert (
            offsets.nplike is self._backend.index_nplike
            and outoffsets.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_NumpyArray_sort_asstrings_uint8",
                self.dtype.type,
                self._data.dtype.type,
                offsets._data.dtype.type,
                outoffsets.dtype.type,
            ](
                out,
                self._data,
                offsets.data,
                offsets.length,
                outoffsets.data,
                True,
                False,
            )
        )

        outlength = ak.index.Index64.empty(1, self._backend.index_nplike)
        nextoffsets = ak.index.Index64.empty(offsets.length, self._backend.index_nplike)
        assert (
            outoffsets.nplike is self._backend.index_nplike
            and nextoffsets.nplike is self._backend.index_nplike
            and outlength.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_NumpyArray_unique_strings",
                self.dtype.type,
                outoffsets.dtype.type,
                nextoffsets.dtype.type,
                outlength.dtype.type,
            ](
                out,
                outoffsets.data,
                offsets.length,
                nextoffsets.data,
                outlength.data,
            )
        )
        out2 = NumpyArray(out, parameters=self._parameters, backend=self._backend)

        return out2, nextoffsets[: outlength[0]]

    def _numbers_to_type(self, name, including_unknown):
        if (
            self.parameter("__array__") == "char"
            or self.parameter("__array__") == "byte"
        ):
            return self
        else:
            dtype = primitive_to_dtype(name)
            return NumpyArray(
                self._backend.nplike.asarray(self._data, dtype=dtype),
                parameters=self._parameters,
                backend=self._backend,
            )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self.length is not unknown_length and self.length == 0:
            return True
        elif len(self.shape) != 1:
            return self.to_RegularArray()._is_unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        elif not self.is_contiguous:
            return self.to_contiguous()._is_unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:
            out = self._unique(negaxis, starts, parents, outlength)
            if isinstance(out, ak.contents.ListOffsetArray):
                return (
                    out.content.length is not unknown_length
                    and out.content.length == self.length
                )
            else:
                return out.length is not unknown_length and out.length == self.length

    def _unique(self, negaxis, starts, parents, outlength):
        if self.shape[0] == 0:
            return self

        elif len(self.shape) == 0:
            return self

        elif negaxis is None:
            contiguous_self = self.to_contiguous()

            offsets = ak.index.Index64.zeros(2, self._backend.index_nplike)
            offsets[1] = self._data.size
            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = self._backend.nplike.empty(self._data.size, dtype=dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    contiguous_self._data,
                    offsets[1],
                    offsets.data,
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )
            nextlength = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert nextlength.nplike is self._backend.index_nplike
            out = self._backend.index_nplike.unique_values(out)
            nextlength[0] = out.size

            return ak.contents.NumpyArray(
                self._backend.nplike.asarray(out[: nextlength[0]], dtype=self.dtype),
                parameters=None,
                backend=self._backend,
            )

        # axis is not None
        elif len(self.shape) != 1:
            return self.to_RegularArray()._unique(
                negaxis,
                starts,
                parents,
                outlength,
            )
        else:
            parents_length = parents.length
            offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges_length",
                    offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )

            offsets = ak.index.Index64.empty(
                offsets_length[0], self._backend.index_nplike
            )
            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length[0],
                    parents.data,
                    parents_length,
                )
            )

            out = self._backend.nplike.empty(self.length, dtype=self.dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sort",
                    out.dtype.type,
                    self._data.dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.data,
                    offsets_length[0],
                    parents_length,
                    True,
                    False,
                )
            )

            nextoffsets = ak.index.Index64.empty(
                offsets.length, self._backend.index_nplike
            )
            assert (
                offsets.nplike is self._backend.index_nplike
                and nextoffsets.nplike is self._backend.index_nplike
            )
            if out.dtype == np.bool_:
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_unique_ranges_bool",
                        out.dtype.type,
                        offsets.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        out,
                        offsets.data,
                        offsets.length,
                        nextoffsets.data,
                    )
                )
            else:
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_unique_ranges",
                        out.dtype.type,
                        offsets.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        out,
                        offsets.data,
                        offsets.length,
                        nextoffsets.data,
                    )
                )

            outoffsets = ak.index.Index64.empty(
                starts.length + 1, self._backend.index_nplike
            )

            assert (
                outoffsets.nplike is self._backend.index_nplike
                and nextoffsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_unique_offsets",
                    outoffsets.dtype.type,
                    nextoffsets.dtype.type,
                    starts.dtype.type,
                ](
                    outoffsets.data,
                    nextoffsets.length,
                    nextoffsets.data,
                    starts.data,
                    starts.length,
                )
            )

            return ak.contents.ListOffsetArray(
                outoffsets, ak.contents.NumpyArray(out), parameters=self._parameters
            )

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        if len(self.shape) != 1:
            return self.to_RegularArray()._argsort_next(
                negaxis, starts, shifts, parents, outlength, ascending, stable
            )
        elif not self.is_contiguous:
            return self.to_contiguous()._argsort_next(
                negaxis, starts, shifts, parents, outlength, ascending, stable
            )
        else:
            parents_length = parents.length
            _offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                _offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges_length",
                    _offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    _offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )
            offsets_length = self._backend.index_nplike.index_as_shape_item(
                _offsets_length[0]
            )

            offsets = ak.index.Index64.empty(offsets_length, self._backend.index_nplike)
            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length,
                    parents.data,
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            nextcarry = ak.index.Index64.empty(self.length, self._backend.index_nplike)
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_argsort",
                    nextcarry.dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    nextcarry.data,
                    self._data,
                    self.length,
                    offsets.data,
                    offsets_length,
                    ascending,
                    stable,
                )
            )

            if shifts is not None:
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and shifts.nplike is self._backend.index_nplike
                    and offsets.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and starts.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_NumpyArray_rearrange_shifted",
                        nextcarry.dtype.type,
                        shifts.dtype.type,
                        offsets.dtype.type,
                        parents.dtype.type,
                        starts.dtype.type,
                    ](
                        nextcarry.data,
                        shifts.data,
                        shifts.length,
                        offsets.data,
                        offsets_length,
                        parents.data,
                        starts.data,
                    )
                )
            out = NumpyArray(nextcarry.data, parameters=None, backend=self._backend)
            return out

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        if len(self.shape) != 1:
            return self.to_RegularArray()._sort_next(
                negaxis, starts, parents, outlength, ascending, stable
            )
        elif not self.is_contiguous:
            return self.to_contiguous()._sort_next(
                negaxis, starts, parents, outlength, ascending, stable
            )

        else:
            parents_length = parents.length
            _offsets_length = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                _offsets_length.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges_length",
                    _offsets_length.dtype.type,
                    parents.dtype.type,
                ](
                    _offsets_length.data,
                    parents.data,
                    parents_length,
                )
            )
            offsets_length = self._backend.index_nplike.index_as_shape_item(
                _offsets_length[0]
            )

            offsets = ak.index.Index64.empty(offsets_length, self._backend.index_nplike)

            assert (
                offsets.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sorting_ranges",
                    offsets.dtype.type,
                    parents.dtype.type,
                ](
                    offsets.data,
                    offsets_length,
                    parents.data,
                    parents_length,
                )
            )

            dtype = (
                np.dtype(np.int64)
                if self._data.dtype.kind.upper() == "M"
                else self._data.dtype
            )
            out = self._backend.nplike.empty(self.length, dtype=dtype)
            assert offsets.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sort",
                    dtype.type,
                    dtype.type,
                    offsets.dtype.type,
                ](
                    out,
                    self._data,
                    self.shape[0],
                    offsets.data,
                    offsets_length,
                    parents_length,
                    ascending,
                    stable,
                )
            )
            return ak.contents.NumpyArray(
                self._backend.nplike.asarray(out, dtype=self.dtype),
                parameters=None,
                backend=self._backend,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif len(self.shape) <= 1:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        else:
            return self.to_RegularArray()._combinations(
                n, replacement, recordlookup, parameters, axis, depth
            )

    def _reduce_next(
        self,
        reducer,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        mask,
        keepdims,
        behavior,
    ):
        if self._data.ndim > 1:
            return self.to_RegularArray()._reduce_next(
                reducer,
                negaxis,
                starts,
                shifts,
                parents,
                outlength,
                mask,
                keepdims,
                behavior,
            )
        elif not self.is_contiguous:
            return self.to_contiguous()._reduce_next(
                reducer,
                negaxis,
                starts,
                shifts,
                parents,
                outlength,
                mask,
                keepdims,
                behavior,
            )

        # Yes, we've just tested these, but we need to be explicit that they are invariants
        assert self.is_contiguous
        assert self._data.ndim == 1

        out = reducer.apply(self, parents, starts, shifts, outlength)

        if mask:
            outmask = ak.index.Index8.empty(outlength, self._backend.index_nplike)
            assert (
                outmask.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
                    outmask.dtype.type,
                    parents.dtype.type,
                ](
                    outmask.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            out = ak.contents.ByteMaskedArray(outmask, out, False, parameters=None)

        if keepdims:
            out = ak.contents.RegularArray(out, 1, self.length, parameters=None)

        return out

    def _validity_error(self, path):
        if len(self.shape) == 0:
            return f"at {path} ({type(self)!r}): shape is zero-dimensional"
        for i, dim in enumerate(self.shape):
            if dim < 0:
                return f"at {path} ({type(self)!r}): shape[{i}] < 0"
        for i, stride in enumerate(self.strides):
            if stride % self.dtype.itemsize != 0:
                return f"at {path} ({type(self)!r}): shape[{i}] % itemsize != 0"
        return ""

    def _pad_none(self, target, axis, depth, clip):
        if len(self.shape) == 0:
            raise ValueError("cannot apply ak.pad_none to a scalar")
        elif len(self.shape) > 1:
            return self.to_RegularArray()._pad_none(target, axis, depth, clip)
        elif not self.is_contiguous:
            return self.to_contiguous()._pad_none(target, axis, depth, clip)
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 != depth:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        if not clip:
            if target < self.length:
                return self
            else:
                return self._pad_none(target, axis, depth, clip=True)
        else:
            return self._pad_none_axis0(target, clip=True)

    def _nbytes_part(self):
        return self.data.nbytes

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        if self._data.ndim != 1:
            return self.to_RegularArray()._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        nparray = self._raw(numpy)
        storage_type = pyarrow.from_numpy_dtype(nparray.dtype)

        if issubclass(nparray.dtype.type, (bool, np.bool_)):
            nparray = numpy.packbits(nparray, bitorder="little")

        return pyarrow.Array.from_buffers(
            ak._connect.pyarrow.to_awkwardarrow_type(
                storage_type,
                options["extensionarray"],
                options["record_is_scalar"],
                mask_node,
                self,
            ),
            length,
            [
                ak._connect.pyarrow.to_validbits(validbytes),
                ak._connect.pyarrow.to_length(nparray, length),
            ],
            null_count=ak._connect.pyarrow.to_null_count(
                validbytes, options["count_nulls"]
            ),
        )

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        cupy = Cupy.instance()
        from cudf.core.column.column import as_column

        assert self._backend.nplike.known_data
        data = as_column(self._data)
        if mask is not None:
            m = cupy.packbits(cupy.asarray(mask), bitorder="little")
            if m.nbytes % 64:
                m = cupy.resize(m, ((m.nbytes // 64) + 1) * 64)
            m = cudf.core.buffer.as_buffer(m)
            data.set_base_data(m)
        return data

    def _to_backend_array(self, allow_missing, backend):
        return to_nplike(self.data, backend.nplike, from_nplike=self._backend.nplike)

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        if options["keepdims"]:
            shape = (1,) * (self._data.ndim - 1) + (-1,)
        else:
            shape = (-1,)
        return [
            ak.contents.NumpyArray(
                backend.nplike.reshape(self._raw(backend.nplike), shape),
                backend=backend,
            )
        ]

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if self._data.ndim != 1 and options["numpy_to_regular"]:
            return self.to_RegularArray()._recursively_apply(
                action, depth, depth_context, lateral_context, options
            )

        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return NumpyArray(
                        self._data, parameters=None, backend=self._backend
                    )

        else:

            def continuation():
                pass

        result = action(
            self,
            depth=depth,
            depth_context=depth_context,
            lateral_context=lateral_context,
            continuation=continuation,
            backend=self._backend,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def to_packed(self, recursive: bool = True) -> Self:
        return self.to_contiguous().to_RegularArray()

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        if self.parameter("__array__") == "byte":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            if convert_bytes is None:
                return ak._util.tobytes(self._data)
            else:
                return convert_bytes(ak._util.tobytes(self._data))

        elif self.parameter("__array__") == "char":
            return ak._util.tobytes(self._data).decode(errors="surrogateescape")

        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        if json_conversions is not None:
            complex_real_string = json_conversions["complex_real_string"]
            complex_imag_string = json_conversions["complex_imag_string"]
            if complex_real_string is not None:
                if issubclass(self.dtype.type, np.complexfloating):
                    return ak.contents.RecordArray(
                        [
                            ak.contents.NumpyArray(
                                self._data.real, backend=self._backend
                            ),
                            ak.contents.NumpyArray(
                                self._data.imag, backend=self._backend
                            ),
                        ],
                        [complex_real_string, complex_imag_string],
                        self.length,
                        parameters=self._parameters,
                        backend=self._backend,
                    )._to_list(behavior, json_conversions)

        out = self._data.tolist()

        if json_conversions is not None:
            nan_string = json_conversions["nan_string"]
            if nan_string is not None:
                for i in self._backend.nplike.nonzero(
                    self._backend.nplike.isnan(self._data)
                )[0]:
                    out[i] = nan_string

            posinf_string = json_conversions["posinf_string"]
            if posinf_string is not None:
                for i in self._backend.nplike.nonzero(self._data == np.inf)[0]:
                    out[i] = posinf_string

            neginf_string = json_conversions["neginf_string"]
            if neginf_string is not None:
                for i in self._backend.nplike.nonzero(self._data == -np.inf)[0]:
                    out[i] = neginf_string

        return out

    def _to_backend(self, backend: Backend) -> Self:
        return NumpyArray(
            self._raw(backend.nplike),
            parameters=self._parameters,
            backend=backend,
        )

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return self._is_equal_to_generic(other, all_parameters) and (
            not numpyarray
            # dtypes agree
            or (
                self.dtype == other.dtype
                # Contents agree
                and (
                    not self._backend.nplike.known_data
                    or self._backend.nplike.array_equal(self.data, other.data)
                )
                # Shapes agree
                and all(
                    x is unknown_length or y is unknown_length or x == y
                    for x, y in zip(self.shape, other.shape)
                )
            )
        )

    def _to_regular_primitive(self) -> ak.contents.RegularArray:
        # A length-1 slice in each dimension
        index = tuple([slice(None, 1)] * len(self.shape))
        # Broadcast this trivial slice to the true dimensions (zero-copy)
        new_data = self.backend.nplike.broadcast_to(self._data[index], self.shape)
        # Convert contiguous array to `RegularArray`
        return NumpyArray(
            new_data, backend=self.backend, parameters=self.parameters
        ).to_RegularArray()
