# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._layout import maybe_posaxis
from awkward._meta.emptymeta import EmptyMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.shape import ShapeItem
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
from awkward.forms.emptyform import EmptyForm
from awkward.forms.form import Form, FormKeyPathT
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

numpy = Numpy.instance()

np = NumpyMetadata.instance()


@final
class EmptyArray(EmptyMeta, Content):
    """
    An EmptyArray is used whenever an array's type is not known because it is empty
    (such as data from #ak.ArrayBuilder without enough sample points to resolve the
    type).

    Unlike all other Content subclasses, EmptyArray cannot contain any parameters
    (parameter values are always None).

    EmptyArray has no equivalent in Apache Arrow.

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class EmptyArray(Content):
            def __init__(self):
                pass

            def __len__(self):
                return 0

            def __getitem__(self, where):
                if isinstance(where, int):
                    assert False

                elif isinstance(where, slice) and where.step is None:
                    return EmptyArray()

                elif isinstance(where, str):
                    raise ValueError("field " + repr(where) + " not found")

                else:
                    raise AssertionError(where)
    """

    def __init__(self, *, parameters=None, backend=None):
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        if backend is None:
            backend = NumpyBackend.instance()
        self._init(parameters, backend)

    form_cls: Final = EmptyForm

    def copy(
        self,
        *,
        parameters=UNSET,
        backend=UNSET,
    ):
        if not (parameters is UNSET or parameters is None or len(parameters) == 0):
            raise TypeError(f"{type(self).__name__} cannot contain parameters")
        return EmptyArray(
            backend=self._backend if backend is UNSET else backend,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    @classmethod
    def simplified(cls, *, parameters=None, backend=None):
        if not (parameters is None or len(parameters) == 0):
            raise TypeError(f"{cls.__name__} cannot contain parameters")
        return cls(backend=backend)

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> EmptyForm:
        return self.form_cls(form_key=getkey(self))

    def _form_with_key_path(self, path: FormKeyPathT) -> EmptyForm:
        return self.form_cls(form_key=repr(path))

    def _to_buffers(
        self,
        form: Form,
        getkey: Callable[[Content, Form, str], str],
        container: MutableMapping[str, ArrayLike],
        backend: Backend,
        byteorder: str,
    ):
        assert isinstance(form, self.form_cls)

    def _to_typetracer(self, forget_length: bool) -> Self:
        return EmptyArray(
            backend=TypeTracerBackend.instance(),
        )

    def _touch_data(self, recursive: bool):
        pass

    def _touch_shape(self, recursive: bool):
        pass

    @property
    def length(self) -> ShapeItem:
        return 0

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        extra = self._repr_extra(indent + "    ")
        if len(extra) == 0:
            return indent + pre + "<EmptyArray len='0'/>" + post
        else:
            out = [indent, pre, "<EmptyArray len='0'>"]
            out.extend(extra)
            out.append("\n" + indent + "</EmptyArray>")
            out.append(post)
            return "".join(out)

    def to_NumpyArray(self, dtype, backend=None):
        backend = backend or self._backend
        return ak.contents.NumpyArray(
            backend.nplike.empty(0, dtype=dtype),
            parameters=self._parameters,
            backend=backend,
        )

    def __iter__(self):
        return iter([])

    def _getitem_nothing(self):
        return self

    def _is_getitem_at_placeholder(self) -> bool:
        return False

    def _is_getitem_at_virtual(self) -> bool:
        return False

    def _getitem_at(self, where: IndexType):
        raise ak._errors.index_error(self, where, "array is empty")

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        return self

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

    def _carry(self, carry: Index, allow_lazy: bool) -> EmptyArray:
        assert isinstance(carry, ak.index.Index)

        if not carry.nplike.known_data or carry.length == 0:
            return self
        else:
            raise ak._errors.index_error(self, carry.data, "array is empty")

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        raise ak._errors.index_error(
            self,
            ak.contents.ListArray(
                slicestarts, slicestops, slicecontent, parameters=None
            ),
            "too many jagged slice dimensions for array",
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
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, slice):
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.index.Index64):
            if not head.nplike.known_data or head.length == 0:
                return self
            else:
                raise ak._errors.index_error(self, head.data, "array is empty")

        elif isinstance(head, ak.contents.ListOffsetArray):
            raise ak._errors.index_error(self, head, "array is empty")

        elif isinstance(head, ak.contents.IndexedOptionArray):
            raise ak._errors.index_error(self, head, "array is empty")

        else:
            raise AssertionError(repr(head))

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError(self, "axis=0 not allowed for flatten")
        else:
            offsets = ak.index.Index64.zeros(1, nplike=self._backend.index_nplike)
            return (
                offsets,
                EmptyArray(backend=self._backend),
            )

    def _mergeable_next(self, other: Content, mergebool: bool) -> bool:
        return True

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self
        elif len(others) == 1:
            return others[0]
        else:
            return others[0]._mergemany(others[1:])

    def _fill_none(self, value: Content) -> Content:
        return EmptyArray(backend=self._backend)

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return ak.contents.NumpyArray(
                self._backend.nplike.empty(0, dtype=np.int64),
                parameters=None,
                backend=self._backend,
            )
        else:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

    def _numbers_to_type(self, name, including_unknown):
        if including_unknown:
            return self.to_NumpyArray(ak.types.numpytype.primitive_to_dtype(name))
        else:
            return self

    def _is_unique(self, negaxis, starts, parents, outlength):
        return True

    def _unique(self, negaxis, starts, parents, outlength):
        return self

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        as_numpy = self.to_NumpyArray(np.float64)
        return as_numpy._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        return self

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ak.contents.EmptyArray(backend=self._backend)

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
        as_numpy = self.to_NumpyArray(reducer.preferred_dtype)
        return as_numpy._reduce_next(
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

    def _validity_error(self, path):
        return ""

    def _nbytes_part(self):
        return 0

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 != depth:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        else:
            return self._pad_none_axis0(target, True)

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        if options["emptyarray_to"] is None:
            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    storage_type=pyarrow.null(),
                    use_extensionarray=options["extensionarray"],
                    record_is_scalar=options["record_is_scalar"],
                    mask=mask_node,
                    node=self,
                    is_nonnullable_nulltype=mask_node is None,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                ],
                null_count=length,
            )

        else:
            dtype = np.dtype(options["emptyarray_to"])
            next = ak.contents.NumpyArray(
                numpy.empty(length, dtype=dtype),
                parameters=self._parameters,
                backend=self._backend,
            )
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        dtype = np.dtype("float64")
        next = ak.contents.NumpyArray(
            numpy.empty(length, dtype=dtype),
            parameters=self._parameters,
            backend=self._backend,
        )
        return next._to_cudf(cudf, None, 0)

    @classmethod
    def _arrow_needs_option_type(cls):
        return True  # This overrides Content._arrow_needs_option_type

    def _to_backend_array(self, allow_missing, backend):
        return backend.nplike.empty(0, dtype=np.float64)

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        return [self]

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if options["return_array"]:

            def continuation():
                if options["keep_parameters"]:
                    return self
                else:
                    return EmptyArray(backend=self._backend)

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
        return self

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")
        return []

    def _to_backend(self, backend: Backend) -> Self:
        return EmptyArray(backend=backend)

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return self._is_equal_to_generic(other, all_parameters)
