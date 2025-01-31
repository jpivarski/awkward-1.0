# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.listoffsetmeta import ListOffsetMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import TypeTracer, is_unknown_scalar
from awkward._nplikes.virtual import VirtualArray
from awkward._parameters import (
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
from awkward.forms.listoffsetform import ListOffsetForm
from awkward.index import Index, Index64

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class ListOffsetArray(ListOffsetMeta[Content], Content):
    """
    ListOffsetArray describes unequal-length lists (often called a
    "jagged" or "ragged" array). Like #ak.contents.RegularArray, the
    underlying data for all lists are in a contiguous `content`. It is
    subdivided into lists according to an `offsets` buffer, which specifies
    the starting and stopping index of each list.

    The `offsets` must have at least length 1 (corresponding to an empty array),
    but it need not start with `0` or include all of the `content`. Just as
    #ak.contents.RegularArray can have unreachable `content` if it is not
    an integer multiple of `size`, a ListOffsetArray can have unreachable
    content before the start of the first list and after the end of the last list.

    Like #ak.contents.RegularArray and #ak.contents.ListArray, a ListOffsetArray can
    represent strings if its `__array__` parameter is `"string"` (UTF-8 assumed) or
    `"bytestring"` (no encoding assumed) and it contains an #ak.contents.NumpyArray
    of `dtype=np.uint8` whose `__array__` parameter is `"char"` (UTF-8 assumed) or
    `"byte"` (no encoding assumed).

    ListOffsetArray corresponds to Apache Arrow
    [List type](https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout).

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class ListOffsetArray(Content):
            def __init__(self, offsets, content):
                assert isinstance(offsets, (Index32, IndexU32, Index64))
                assert isinstance(content, Content)
                assert len(offsets) != 0
                for i in range(len(offsets) - 1):
                    start = offsets[i]
                    stop = offsets[i + 1]
                    if start != stop:
                        assert start < stop  # i.e. start <= stop
                        assert start >= 0
                        assert stop <= len(content)
                self.offsets = offsets
                self.content = content

            def __len__(self):
                return len(self.offsets) - 1

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    return self.content[self.offsets[where] : self.offsets[where + 1]]

                elif isinstance(where, slice) and where.step is None:
                    offsets = self.offsets[where.start : where.stop + 1]
                    if len(offsets) == 0:
                        offsets = [0]
                    return ListOffsetArray(offsets, self.content)

                elif isinstance(where, str):
                    return ListOffsetArray(self.offsets, self.content[where])

                else:
                    raise AssertionError(where)
    """

    def __init__(self, offsets, content, *, parameters=None):
        if not isinstance(offsets, Index) and offsets.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                f"{type(self).__name__} 'offsets' must be an Index with dtype in (int32, uint32, int64), "
                f"not {offsets!r}"
            )
        if not isinstance(content, Content):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Content subtype, not {content!r}"
            )
        if (
            content.backend.index_nplike.known_data
            and offsets.length is not unknown_length
            and offsets.length == 0
        ):
            raise ValueError(
                f"{type(self).__name__} len(offsets) ({offsets.length}) must be >= 1"
            )

        if parameters is not None and parameters.get("__array__") == "string":
            if not content.is_numpy or not content.parameter("__array__") == "char":
                raise ValueError(
                    f"{type(self).__name__} is a string, so its 'content' must be uint8 NumpyArray of char, not {content!r}"
                )
        if parameters is not None and parameters.get("__array__") == "bytestring":
            if not content.is_numpy or not content.parameter("__array__") == "byte":
                raise ValueError(
                    f"{type(self).__name__} is a bytestring, so its 'content' must be uint8 NumpyArray of byte, not {content!r}"
                )

        assert offsets.nplike is content.backend.index_nplike

        self._offsets = offsets
        self._content = content
        self._init(parameters, content.backend)

    @property
    def offsets(self) -> Index:
        return self._offsets

    form_cls: Final = ListOffsetForm

    def copy(self, offsets=UNSET, content=UNSET, *, parameters=UNSET):
        return ListOffsetArray(
            self._offsets if offsets is UNSET else offsets,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            offsets=copy.deepcopy(self._offsets, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, offsets, content, *, parameters=None):
        return cls(offsets, content, parameters=parameters)

    @property
    def starts(self) -> Index:
        return self._offsets[:-1]

    @property
    def stops(self):
        return self._offsets[1:]

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> ListOffsetForm:
        form_key = getkey(self)
        return self.form_cls(
            self._offsets.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> ListOffsetForm:
        return self.form_cls(
            self._offsets.form,
            self._content._form_with_key_path((*path, None)),
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
        key = getkey(self, form, "offsets")
        container[key] = ak._util.native_to_byteorder(
            self._offsets.raw(backend.index_nplike), byteorder
        )
        self._content._to_buffers(form.content, getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        offsets = self._offsets.to_nplike(TypeTracer.instance())
        return ListOffsetArray(
            offsets.forget_length() if forget_length else offsets,
            self._content._to_typetracer(forget_length),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        self._offsets._touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        self._offsets._touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._offsets.length - 1

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListOffsetArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._offsets._repr(indent + "    ", "<offsets>", "</offsets>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ListOffsetArray>")
        out.append(post)
        return "".join(out)

    def to_ListOffsetArray64(self, start_at_zero: bool = False) -> ListOffsetArray:
        known_starts_at_zero = (
            self._backend.index_nplike.known_data and self._offsets[0] == 0
        )
        if start_at_zero and not known_starts_at_zero:
            offsets = Index64(
                self._offsets.data - self._offsets[0],
                nplike=self._backend.index_nplike,
            )
            return ListOffsetArray(
                offsets, self._content[self._offsets[0] :], parameters=self._parameters
            )
        else:
            return ListOffsetArray(
                self._offsets.to64(), self._content, parameters=self._parameters
            )

    def to_RegularArray(self):
        start, stop = (
            self._offsets[0],
            self._offsets[
                self._backend.index_nplike.shape_item_as_index(self._offsets.length - 1)
            ],
        )
        content = self._content._getitem_range(start, stop)
        _size = Index64.empty(1, self._backend.index_nplike)
        assert (
            _size.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListOffsetArray_toRegularArray",
                _size.dtype.type,
                self._offsets.dtype.type,
            ](
                _size.data,
                self._offsets.data,
                self._offsets.length,
            )
        )
        size = self._backend.index_nplike.index_as_shape_item(_size[0])
        length = self._offsets.length - 1
        return ak.contents.RegularArray(
            content, size, length, parameters=self._parameters
        )

    def _getitem_nothing(self):
        return self._content._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder = isinstance(self._offsets.data, PlaceholderArray)
        return is_placeholder or self._content._is_getitem_at_placeholder()

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual = (
            isinstance(self._offsets.data, VirtualArray)
            and not self._offsets.data.is_materialized
        )
        return is_virtual or self._content._is_getitem_at_virtual()

    def _getitem_at(self, where: IndexType):
        # Wrap `where` by length
        if not is_unknown_scalar(where) and where < 0:
            length_index = self._backend.index_nplike.shape_item_as_index(self.length)
            where += length_index
        # Validate `where`
        if not (
            is_unknown_scalar(where)
            or self.length is unknown_length
            or (0 <= where < self.length)
        ):
            raise ak._errors.index_error(self, where)
        start, stop = self._offsets[where], self._offsets[where + 1]
        return self._content._getitem_range(start, stop)

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        offsets = self._offsets[start : stop + 1]
        if offsets.length is not unknown_length and offsets.length == 0:
            offsets = Index(
                self._backend.index_nplike.zeros(1, dtype=self._offsets.dtype),
                nplike=self._backend.index_nplike,
            )
        return ListOffsetArray(offsets, self._content, parameters=self._parameters)

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ListOffsetArray(
            self._offsets,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)

        try:
            nextstarts = self.starts[carry.data]
            nextstops = self.stops[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return ak.contents.ListArray(
            nextstarts, nextstops, self._content, parameters=self._parameters
        )

    def _compact_offsets64(self, start_at_zero: bool) -> Index64:
        if not start_at_zero or (
            self._backend.index_nplike.known_data and self._offsets[0] == 0
        ):
            return self._offsets
        else:
            return Index64(
                self._offsets.data - self._offsets[0],
                nplike=self._backend.index_nplike,
            )

    def _broadcast_tooffsets64(self, offsets: Index) -> ListOffsetArray:
        self._touch_data(recursive=False)
        offsets._touch_data()

        index_nplike = self._backend.index_nplike
        assert offsets.nplike is index_nplike
        if offsets.length is not unknown_length and offsets.length == 0:
            raise AssertionError(
                "broadcast_tooffsets64 can only be used with non-empty offsets"
            )
        elif index_nplike.known_data and offsets[0] != 0:
            raise AssertionError(
                f"broadcast_tooffsets64 can only be used with offsets that start at 0, not {offsets[0]}"
            )
        elif (
            offsets.length is not unknown_length
            and self._offsets.length is not unknown_length
            and offsets.length != self._offsets.length
        ):
            raise AssertionError(
                f"cannot broadcast RegularArray of length {self.length} to length {offsets.length - 1}"
            )

        # Check whether we need to slice the content, shift our offsets
        this_start = self._offsets[0]
        this_zero_offsets = self._offsets.data
        if index_nplike.known_data and this_start == 0:
            next_content = self._content
        else:
            this_zero_offsets = this_zero_offsets - this_start
            next_content = self._content[this_start:]

        if index_nplike.known_data and not index_nplike.array_equal(
            this_zero_offsets, offsets.data
        ):
            raise ValueError("cannot broadcast nested list")

        return ListOffsetArray(
            offsets, next_content[: offsets[-1]], parameters=self._parameters
        )

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        out = ak.contents.ListArray(
            self.starts, self.stops, self._content, parameters=self._parameters
        )
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        if head is NO_HEAD:
            return self

        elif is_integer_like(head):
            assert advanced is None
            lenstarts = self._offsets.length - 1
            starts, stops = self.starts, self.stops
            nexthead, nexttail = ak._slicing.head_tail(tail)
            nextcarry = Index64.empty(lenstarts, self._backend.index_nplike)

            assert (
                nextcarry.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    starts.dtype.type,
                    stops.dtype.type,
                ](
                    nextcarry.data,
                    starts.data,
                    stops.data,
                    lenstarts,
                    head,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._slicing.head_tail(tail)
            lenstarts = self._offsets.length - 1
            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            carrylength = Index64.empty(1, self._backend.index_nplike)
            assert (
                carrylength.nplike is self._backend.index_nplike
                and self.starts.nplike is self._backend.index_nplike
                and self.stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_next_range_carrylength",
                    carrylength.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    carrylength.data,
                    self.starts.data,
                    self.stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                slicer=head,
            )

            if self._starts.dtype == "int64":
                nextoffsets = Index64.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            elif self._starts.dtype == "int32":
                nextoffsets = ak.index.Index32.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            elif self._starts.dtype == "uint32":
                nextoffsets = ak.index.IndexU32.empty(
                    lenstarts + 1, nplike=self._backend.index_nplike
                )
            nextcarry = Index64.empty(carrylength[0], self._backend.index_nplike)

            assert (
                nextoffsets.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and self.starts.nplike is self._backend.index_nplike
                and self.stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self.starts.dtype.type,
                    self.stops.dtype.type,
                ](
                    nextoffsets.data,
                    nextcarry.data,
                    self.starts.data,
                    self.stops.data,
                    lenstarts,
                    start,
                    stop,
                    step,
                ),
                slicer=head,
            )

            nextcontent = self._content._carry(nextcarry, True)

            if advanced is None or (
                advanced.length is not unknown_length and advanced.length == 0
            ):
                return ak.contents.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    parameters=self._parameters,
                )

            else:
                total = Index64.empty(1, self._backend.index_nplike)
                assert (
                    total.nplike is self._backend.index_nplike
                    and nextoffsets.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_range_counts",
                        total.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        total.data,
                        nextoffsets.data,
                        lenstarts,
                    ),
                    slicer=head,
                )

                nextadvanced = Index64.empty(total[0], self._backend.index_nplike)
                assert (
                    nextadvanced.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                    and nextoffsets.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        nextoffsets.dtype.type,
                    ](
                        nextadvanced.data,
                        advanced.data,
                        nextoffsets.data,
                        lenstarts,
                    ),
                    slicer=head,
                )

                return ak.contents.ListOffsetArray(
                    nextoffsets,
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    parameters=self._parameters,
                )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, Index64):
            nexthead, nexttail = ak._slicing.head_tail(tail)
            flathead = self._backend.index_nplike.reshape(
                self._backend.index_nplike.asarray(head.data), (-1,)
            )
            lenstarts = self.starts.length
            regular_flathead = Index64(flathead)
            if advanced is None or (
                advanced.length is not unknown_length and advanced.length == 0
            ):
                nextcarry = Index64.empty(
                    lenstarts * flathead.length, self._backend.index_nplike
                )
                nextadvanced = Index64.empty(
                    lenstarts * flathead.length, self._backend.index_nplike
                )
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self.starts.data,
                        self.stops.data,
                        regular_flathead.data,
                        lenstarts,
                        regular_flathead.length,
                        self._content.length,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,), self.length)
                    )
                else:
                    return out

            else:
                nextcarry = Index64.empty(self.length, self._backend.index_nplike)
                nextadvanced = Index64.empty(self.length, self._backend.index_nplike)
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and self.starts.nplike is self._backend.index_nplike
                    and self.stops.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self.starts.dtype.type,
                        self.stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self.starts.data,
                        self.stops.data,
                        regular_flathead.data,
                        advanced.data,
                        lenstarts,
                        self._content.length,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

        elif isinstance(head, ak.contents.ListOffsetArray):
            listarray = ak.contents.ListArray(
                self.starts, self.stops, self._content, parameters=self._parameters
            )
            return listarray._getitem_next(head, tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError("axis=0 not allowed for flatten")

        elif posaxis is not None and posaxis + 1 == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ValueError(
                    "array of strings cannot be directly flattened. "
                    'To flatten this array, drop the `"__array__"="string"` parameter using '
                    "`ak.enforce_type`, `ak.with_parameter`, or `ak.without_parameters`/"
                )
            listoffsetarray = self.to_ListOffsetArray64(True)
            stop = listoffsetarray.offsets[-1]
            content = listoffsetarray.content._getitem_range(0, stop)
            return (listoffsetarray.offsets, content)

        else:
            inneroffsets, flattened = self._content._offsets_and_flattened(
                axis, depth + 1
            )
            offsets = Index64.zeros(
                0,
                nplike=self._backend.index_nplike,
                dtype=np.int64,
            )

            if inneroffsets.length is not unknown_length and inneroffsets.length == 0:
                return (
                    offsets,
                    ListOffsetArray(
                        self._offsets, flattened, parameters=self._parameters
                    ),
                )

            elif (
                self._offsets.length is not unknown_length and self._offsets.length == 1
            ):
                tooffsets = Index64([inneroffsets[0]])
                return (
                    offsets,
                    ListOffsetArray(tooffsets, flattened, parameters=self._parameters),
                )

            else:
                tooffsets = Index64.empty(
                    self._offsets.length,
                    self._backend.index_nplike,
                    dtype=np.int64,
                )
                assert (
                    tooffsets.nplike is self._backend.index_nplike
                    and self._offsets.nplike is self._backend.index_nplike
                    and inneroffsets.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_flatten_offsets",
                        tooffsets.dtype.type,
                        self._offsets.dtype.type,
                        inneroffsets.dtype.type,
                    ](
                        tooffsets.data,
                        self._offsets.data,
                        self._offsets.length,
                        inneroffsets.data,
                    )
                )
                return (
                    offsets,
                    ListOffsetArray(tooffsets, flattened, parameters=self._parameters),
                )

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
        elif isinstance(
            other,
            (
                ak.contents.RegularArray,
                ak.contents.ListArray,
                ak.contents.ListOffsetArray,
            ),
        ):
            return self._content._mergeable_next(other.content, mergebool)
        elif isinstance(other, ak.contents.NumpyArray) and len(other.shape) > 1:
            return self._mergeable_next(other._to_regular_primitive(), mergebool)
        else:
            return False

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self
        listarray = ak.contents.ListArray(
            self.starts, self.stops, self._content, parameters=self._parameters
        )
        out = listarray._mergemany(others)

        if all(
            isinstance(x, ListOffsetArray) and x._offsets.dtype == self._offsets.dtype
            for x in others
        ):
            return out.to_ListOffsetArray64(False)
        else:
            return out

    def _fill_none(self, value: Content) -> Content:
        return ListOffsetArray(
            self._offsets, self._content._fill_none(value), parameters=self._parameters
        )

    def _local_index(self, axis, depth):
        index_nplike = self._backend.index_nplike
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif posaxis is not None and posaxis + 1 == depth + 1:
            offsets = self._compact_offsets64(True)
            if self._backend.nplike.known_data:
                innerlength = index_nplike.index_as_shape_item(
                    offsets[index_nplike.shape_item_as_index(offsets.length) - 1]
                )
            else:
                self._touch_data(recursive=False)
                innerlength = unknown_length
            localindex = Index64.empty(innerlength, index_nplike)
            assert localindex.nplike is index_nplike and offsets.nplike is index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListArray_localindex",
                    localindex.dtype.type,
                    offsets.dtype.type,
                ](
                    localindex.data,
                    offsets.data,
                    offsets.length - 1,
                )
            )
            return ak.contents.ListOffsetArray(
                offsets, ak.contents.NumpyArray(localindex.data)
            )
        else:
            return ak.contents.ListOffsetArray(
                self._offsets, self._content._local_index(axis, depth + 1)
            )

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.ListOffsetArray(
            self._offsets,
            self._content._numbers_to_type(name, including_unknown),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._offsets.length - 1 == 0:
            return True

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis is not None and negaxis != depth):
                raise ValueError(
                    "array with strings can only be checked on uniqueness with axis=-1"
                )

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                out, outoffsets = self._content._as_unique_strings(self._offsets)
                out2 = ak.contents.ListOffsetArray(
                    outoffsets, out, parameters=self._parameters
                )
                return out2.length == self.length

        if negaxis is None:
            return self._content._is_unique(negaxis, starts, parents, outlength)

        if not branch and (negaxis == depth):
            return self._content._is_unique(negaxis - 1, starts, parents, outlength)
        else:
            nextlen = self._backend.index_nplike.index_as_shape_item(
                self._offsets[-1] - self._offsets[0]
            )
            nextparents = Index64.empty(nextlen, self._backend.index_nplike)

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )
            starts = self._offsets[:-1]

            return self._content._is_unique(negaxis, starts, nextparents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if self._offsets.length - 1 == 0:
            return self

        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                out, nextoffsets = self._content._as_unique_strings(self._offsets)
                return ak.contents.ListOffsetArray(
                    nextoffsets, out, parameters=self._parameters
                )

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise AxisError("array with strings can only be sorted with axis=-1")

            if self._backend.nplike.known_data and parents.nplike.known_data:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._unique(
                negaxis - 1,
                nextstarts,
                nextparents,
                maxnextparents[0] + 1,
            )

            outcarry = Index64.empty(nextcarry.length, self._backend.index_nplike)
            assert (
                outcarry.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            return ak.contents.ListOffsetArray(
                outcontent._compact_offsets64(True),
                outcontent._content._carry(outcarry, False),
                parameters=self._parameters,
            )

        else:
            nextlen = self._backend.index_nplike.index_as_shape_item(
                self._offsets[-1] - self._offsets[0]
            )
            nextparents = Index64.empty(nextlen, self._backend.index_nplike)

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._unique(
                negaxis,
                self._offsets[:-1],
                nextparents,
                self._offsets.length - 1,
            )

            if negaxis is None or negaxis == depth - 1:
                return outcontent

            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

    def _argsort_next(
        self,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        ascending,
        stable,
    ):
        branch, depth = self.branch_depth

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                nextcarry = Index64.empty(
                    self._offsets.length - 1, self._backend.index_nplike
                )

                self_starts, self_stops = self._offsets[:-1], self._offsets[1:]
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and parents.nplike is self._backend.index_nplike
                    and self._content.backend is self._backend
                    and self_starts.nplike is self._backend.index_nplike
                    and self_stops.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        self_starts.dtype.type,
                        self_stops.dtype.type,
                    ](
                        nextcarry.data,
                        parents.data,
                        parents.length,
                        self._content._data,
                        self_starts.data,
                        self_stops.data,
                        stable,
                        ascending,
                        True,
                    )
                )
                return ak.contents.NumpyArray(
                    nextcarry.data, parameters=None, backend=self._backend
                )

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise AxisError("array with strings can only be sorted with axis=-1")

            if self._backend.nplike.known_data and parents.nplike.known_data:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nummissing = Index64.empty(maxcount, self._backend.index_nplike)
            missing = Index64.empty(self._offsets[-1], self._backend.index_nplike)
            nextshifts = Index64.empty(nextcarry.length, self._backend.index_nplike)
            assert (
                nummissing.nplike is self._backend.index_nplike
                and missing.nplike is self._backend.index_nplike
                and nextshifts.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )

            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                    nummissing.dtype.type,
                    missing.dtype.type,
                    nextshifts.dtype.type,
                    self._offsets.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextcarry.dtype.type,
                ](
                    nummissing.data,
                    missing.data,
                    nextshifts.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                    starts.data,
                    parents.data,
                    maxcount,
                    nextcarry.length,
                    nextcarry.data,
                )
            )

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._argsort_next(
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                nextstarts.length,
                ascending,
                stable,
            )

            outcarry = Index64.empty(nextcarry.length, self._backend.index_nplike)
            assert (
                outcarry.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            out_offsets = self._compact_offsets64(True)
            out = outcontent._carry(outcarry, False)
            return ak.contents.ListOffsetArray(
                out_offsets, out, parameters=self._parameters
            )
        else:
            nextlen = self._backend.index_nplike.index_as_shape_item(
                self._offsets[-1] - self._offsets[0]
            )
            nextparents = Index64.empty(nextlen, self._backend.index_nplike)

            assert (
                nextparents.nplike is self._backend.index_nplike
                and self._offsets.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    self._offsets.length - 1,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._argsort_next(
                negaxis,
                self._offsets[:-1],
                shifts,
                nextparents,
                self._offsets.length - 1,
                ascending,
                stable,
            )
            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        branch, depth = self.branch_depth

        index_nplike = self._backend.index_nplike

        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            if branch or (negaxis != depth):
                raise AxisError("array with strings can only be sorted with axis=-1")

            # FIXME: check validity error

            if isinstance(self._content, ak.contents.NumpyArray):
                nextcarry = Index64.empty(self._offsets.length - 1, index_nplike)

                starts, stops = self._offsets[:-1], self._offsets[1:]
                assert (
                    nextcarry.nplike is index_nplike
                    and parents.nplike is index_nplike
                    and self._content.backend is self._backend
                    and starts.nplike is index_nplike
                    and stops.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_argsort_strings",
                        nextcarry.dtype.type,
                        parents.dtype.type,
                        self._content.dtype.type,
                        starts.dtype.type,
                        stops.dtype.type,
                    ](
                        nextcarry.data,
                        parents.data,
                        parents.length,
                        self._content._data,
                        starts.data,
                        stops.data,
                        stable,
                        ascending,
                        False,
                    )
                )
                return self._carry(nextcarry, False)

        if not branch and (negaxis == depth):
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise AxisError("array with strings can only be sorted with axis=-1")

            if self._backend.nplike.known_data and parents.nplike.known_data:
                assert self._offsets.length - 1 == parents.length

            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._sort_next(
                negaxis - 1,
                nextstarts,
                nextparents,
                maxnextparents + 1,
                ascending,
                stable,
            )

            outcarry = Index64.empty(nextcarry.length, index_nplike)
            assert outcarry.nplike is index_nplike and nextcarry.nplike is index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_local_preparenext_64",
                    outcarry.dtype.type,
                    nextcarry.dtype.type,
                ](
                    outcarry.data,
                    nextcarry.data,
                    nextcarry.length,
                )
            )

            return ak.contents.ListOffsetArray(
                self._compact_offsets64(True),
                outcontent._carry(outcarry, False),
                parameters=self._parameters,
            )
        else:
            nextlen = index_nplike.index_as_shape_item(
                self._offsets[-1] - self._offsets[0]
            )
            nextparents = Index64.empty(nextlen, index_nplike)
            lenstarts = self._offsets.length - 1

            assert (
                nextparents.nplike is index_nplike
                and self._offsets.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    lenstarts,
                )
            )

            trimmed = self._content[self._offsets[0] : self._offsets[-1]]
            outcontent = trimmed._sort_next(
                negaxis,
                self._offsets[:-1],
                nextparents,
                lenstarts,
                ascending,
                stable,
            )
            outoffsets = self._compact_offsets64(True)
            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        index_nplike = self._backend.index_nplike

        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            if (
                self.parameter("__array__") == "string"
                or self.parameter("__array__") == "bytestring"
            ):
                raise ValueError(
                    "ak.combinations does not compute combinations of the characters of a string; please split it into lists"
                )

            starts = self.starts
            stops = self.stops

            _totallen = Index64.empty(1, index_nplike, dtype=np.int64)
            offsets = Index64.empty(
                self.length + 1,
                index_nplike,
                dtype=np.int64,
            )
            assert (
                offsets.nplike is index_nplike
                and starts.nplike is index_nplike
                and stops.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListArray_combinations_length",
                    _totallen.data.dtype.type,
                    offsets.data.dtype.type,
                    starts.data.dtype.type,
                    stops.data.dtype.type,
                ](
                    _totallen.data,
                    offsets.data,
                    n,
                    replacement,
                    starts.data,
                    stops.data,
                    self.length,
                )
            )
            totallen = self._backend.index_nplike.index_as_shape_item(_totallen[0])

            tocarryraw = ak.index.Index.empty(n, dtype=np.intp, nplike=index_nplike)
            tocarry = []

            for i in range(n):
                ptr = Index64.empty(
                    totallen,
                    nplike=index_nplike,
                    dtype=np.int64,
                )
                tocarry.append(ptr)
                if self._backend.nplike.known_data:
                    tocarryraw[i] = ptr.ptr

            toindex = Index64.empty(n, index_nplike, dtype=np.int64)
            fromindex = Index64.empty(n, index_nplike, dtype=np.int64)
            assert (
                toindex.nplike is index_nplike
                and fromindex.nplike is index_nplike
                and starts.nplike is index_nplike
                and stops.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListArray_combinations",
                    np.int64,
                    toindex.data.dtype.type,
                    fromindex.data.dtype.type,
                    starts.data.dtype.type,
                    stops.data.dtype.type,
                ](
                    tocarryraw.data,
                    toindex.data,
                    fromindex.data,
                    n,
                    replacement,
                    starts.data,
                    stops.data,
                    self.length,
                )
            )
            contents = []

            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True))

            recordarray = ak.contents.RecordArray(
                contents,
                recordlookup,
                parameters=parameters,
                backend=self._backend,
            )
            return ak.contents.ListOffsetArray(
                offsets, recordarray, parameters=self._parameters
            )
        else:
            compact = self.to_ListOffsetArray64(True)
            next = compact._content._combinations(
                n, replacement, recordlookup, parameters, axis, depth + 1
            )
            return ak.contents.ListOffsetArray(
                compact.offsets, next, parameters=self._parameters
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
        index_nplike = self._backend.index_nplike

        if self._offsets.dtype != np.dtype(np.int64) or (
            self._offsets.nplike.known_data and self._offsets[0] != 0
        ):
            next = self.to_ListOffsetArray64(True)
            return next._reduce_next(
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

        branch, depth = self.branch_depth
        globalstarts_length = self._offsets.length - 1

        if not branch and negaxis == depth:
            (
                distincts,
                maxcount,
                maxnextparents,
                nextcarry,
                nextparents,
                nextstarts,
            ) = self._rearrange_prepare_next(outlength, parents)

            outstarts = Index64.empty(outlength, index_nplike)
            outstops = Index64.empty(outlength, index_nplike)
            assert (
                outstarts.nplike is index_nplike
                and outstops.nplike is index_nplike
                and distincts.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
                    outstarts.dtype.type,
                    outstops.dtype.type,
                    distincts.dtype.type,
                ](
                    outstarts.data,
                    outstops.data,
                    distincts.data,
                    distincts.length,
                    outlength,
                )
            )

            if reducer.needs_position:
                nextshifts = Index64.empty(nextcarry.length, index_nplike)
                nummissing = Index64.empty(maxcount, index_nplike)
                missing = Index64.empty(
                    index_nplike.index_as_shape_item(self._offsets[-1]),
                    index_nplike,
                )
                assert (
                    nummissing.nplike is index_nplike
                    and missing.nplike is index_nplike
                    and nextshifts.nplike is index_nplike
                    and self._offsets.nplike is index_nplike
                    and starts.nplike is index_nplike
                    and parents.nplike is index_nplike
                    and nextcarry.nplike is index_nplike
                )

                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
                        nummissing.dtype.type,
                        missing.dtype.type,
                        nextshifts.dtype.type,
                        self._offsets.dtype.type,
                        starts.dtype.type,
                        parents.dtype.type,
                        nextcarry.dtype.type,
                    ](
                        nummissing.data,
                        missing.data,
                        nextshifts.data,
                        self._offsets.data,
                        globalstarts_length,
                        starts.data,
                        parents.data,
                        maxcount,
                        nextcarry.length,
                        nextcarry.data,
                    )
                )
            else:
                nextshifts = None

            nextcontent = self._content._carry(nextcarry, False)
            outcontent = nextcontent._reduce_next(
                reducer,
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                maxnextparents + 1,
                mask,
                False,
                behavior,
            )

            out = ak.contents.ListArray(
                outstarts, outstops, outcontent, parameters=None
            )

            if keepdims:
                out = ak.contents.RegularArray(out, 1, self.length, parameters=None)

            return out

        else:
            nextlen = index_nplike.index_as_shape_item(
                self._offsets[-1] - self._offsets[0]
            )
            nextparents = Index64.empty(nextlen, index_nplike)

            # n.b. awkward_ListOffsetArray_reduce_local_nextparents_64 always returns parents that are
            # monotonically increasing (because it is local)
            assert (
                nextparents.nplike is index_nplike
                and self._offsets.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                    self._offsets.dtype.type,
                ](
                    nextparents.data,
                    self._offsets.data,
                    globalstarts_length,
                )
            )

            trimmed = self._content[self.offsets[0] : self.offsets[-1]]
            nextstarts = self.offsets[:-1]

            outcontent = trimmed._reduce_next(
                reducer,
                negaxis,
                nextstarts,
                shifts,
                nextparents,
                globalstarts_length,
                mask,
                keepdims,
                behavior,
            )

            outoffsets = Index64.empty(outlength + 1, index_nplike)
            assert outoffsets.nplike is index_nplike and parents.nplike is index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
                    outoffsets.dtype.type,
                    parents.dtype.type,
                ](
                    outoffsets.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

            # `outcontent` represents *this* layout in the reduction
            # If this layout survives in the reduction (see `if` below), then we want
            # to ensure that we have a ragged list type (unless it's a `keepdims=True` layout)
            if keepdims and depth == negaxis + 1:
                # Don't convert the `RegularArray()` to a `ListOffsetArray`,
                # means this will be broadcastable
                assert outcontent.is_regular
            elif depth >= negaxis + 2:
                # The *only* >1D list types that we can have as direct children
                # are the `is_list` or `is_regular` types; NumpyArray should be
                # converted to `RegularArray`.
                assert outcontent.is_list or outcontent.is_regular
                outcontent = outcontent.to_ListOffsetArray64(False)

            return ak.contents.ListOffsetArray(outoffsets, outcontent, parameters=None)

    def _rearrange_prepare_next(self, outlength, parents):
        index_nplike = self._backend.index_nplike
        nextlen = index_nplike.index_as_shape_item(self._offsets[-1] - self._offsets[0])
        lenstarts = self._offsets.length - 1
        _maxcount = Index64.empty(1, index_nplike)
        offsetscopy = Index64.empty(self.offsets.length, index_nplike)
        assert (
            _maxcount.nplike is index_nplike
            and offsetscopy.nplike is index_nplike
            and self._offsets.nplike is index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
                _maxcount.dtype.type,
                offsetscopy.dtype.type,
                self._offsets.dtype.type,
            ](
                _maxcount.data,
                offsetscopy.data,
                self._offsets.data,
                lenstarts,
            )
        )
        maxcount = index_nplike.index_as_shape_item(_maxcount[0])

        nextcarry = Index64.empty(nextlen, nplike=index_nplike)
        nextparents = Index64.empty(nextlen, nplike=index_nplike)
        _maxnextparents = Index64.empty(1, index_nplike)
        if maxcount is unknown_length or outlength is unknown_length:
            distincts = Index64.empty(unknown_length, index_nplike)
        else:
            distincts = Index64.empty(outlength * maxcount, index_nplike)

        assert (
            _maxnextparents.nplike is index_nplike
            and distincts.nplike is index_nplike
            and self._offsets.nplike is index_nplike
            and offsetscopy.nplike is index_nplike
            and parents.nplike is index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                _maxnextparents.dtype.type,
                distincts.dtype.type,
                self._offsets.dtype.type,
                offsetscopy.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                nextlen,
                _maxnextparents.data,
                distincts.data,
                distincts.length,
                offsetscopy.data,
                self._offsets.data,
                lenstarts,
                parents.data,
                maxcount,
            )
        )

        maxnextparents = index_nplike.index_as_shape_item(_maxnextparents[0])
        nextstarts = Index64.empty(maxnextparents + 1, index_nplike)
        assert nextstarts.nplike is index_nplike and nextparents.nplike is index_nplike
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
                nextstarts.dtype.type,
                nextparents.dtype.type,
            ](
                nextstarts.data,
                nextparents.data,
                nextlen,
            )
        )
        return (
            distincts,
            maxcount,
            maxnextparents,
            nextcarry,
            nextparents,
            nextstarts,
        )

    def _validity_error(self, path):
        if self.offsets.length < 1:
            return f"at {path} ({type(self)!r}): len(offsets) < 1"
        assert (
            self.starts.nplike is self._backend.index_nplike
            and self.stops.nplike is self._backend.index_nplike
        )
        error = self._backend[
            "awkward_ListArray_validity", self.starts.dtype.type, self.stops.dtype.type
        ](
            self.starts.data,
            self.stops.data,
            self.starts.length,
            self._content.length,
        )
        if error.str is not None:
            if error.filename is None:
                filename = ""
            else:
                filename = " (in compiled code: " + error.filename.decode(
                    errors="surrogateescape"
                ).lstrip("\n").lstrip("(")
            message = error.str.decode(errors="surrogateescape")
            return f'at {path} ("{type(self)}"): {message} at i={error.id}{filename}'
        else:
            return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.offsets._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        index_nplike = self._backend.index_nplike

        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        if posaxis is not None and posaxis + 1 == depth + 1:
            if not clip:
                _tolength = Index64.empty(1, index_nplike)
                offsets_ = Index64.empty(self._offsets.length, index_nplike)
                assert (
                    offsets_.nplike is index_nplike
                    and self._offsets.nplike is index_nplike
                    and _tolength.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_length_axis1",
                        offsets_.dtype.type,
                        self._offsets.dtype.type,
                        _tolength.dtype.type,
                    ](
                        offsets_.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                        _tolength.data,
                    )
                )
                tolength = index_nplike.index_as_shape_item(_tolength[0])
                outindex = Index64.empty(tolength, index_nplike)
                assert (
                    outindex.nplike is index_nplike
                    and self._offsets.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    outindex, self._content, parameters=self._parameters
                )
                return ak.contents.ListOffsetArray(
                    offsets_, next, parameters=self._parameters
                )
            else:
                starts_ = Index64.empty(
                    self._offsets.length - 1,
                    index_nplike,
                )
                stops_ = Index64.empty(
                    self._offsets.length - 1,
                    index_nplike,
                )
                assert starts_.nplike is index_nplike and stops_.nplike is index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_index_rpad_and_clip_axis1",
                        starts_.dtype.type,
                        stops_.dtype.type,
                    ](
                        starts_.data,
                        stops_.data,
                        target,
                        starts_.length,
                    )
                )

                outindex = Index64.empty(
                    target * (self._offsets.length - 1),
                    index_nplike,
                )
                assert (
                    outindex.nplike is index_nplike
                    and self._offsets.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListOffsetArray_rpad_and_clip_axis1",
                        outindex.dtype.type,
                        self._offsets.dtype.type,
                    ](
                        outindex.data,
                        self._offsets.data,
                        self._offsets.length - 1,
                        target,
                    )
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    outindex, self._content, parameters=self._parameters
                )
                return ak.contents.RegularArray(
                    next,
                    target,
                    self.length,
                    parameters=self._parameters,
                )
        else:
            return ak.contents.ListOffsetArray(
                self._offsets,
                self._content._pad_none(target, axis, depth + 1, clip),
                parameters=self._parameters,
            )

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        is_string = self.parameter("__array__") == "string"
        is_bytestring = self.parameter("__array__") == "bytestring"
        if is_string:
            downsize = options["string_to32"]
        elif is_bytestring:
            downsize = options["bytestring_to32"]
        else:
            downsize = options["list_to32"]
        npoffsets = self._offsets.raw(numpy)
        akcontent = self._content[npoffsets[0] : npoffsets[length]]
        if len(npoffsets) > length + 1:
            npoffsets = npoffsets[: length + 1]
        if npoffsets[0] != 0:
            npoffsets = npoffsets - npoffsets[0]

        # ArrowNotImplementedError: Lists with non-zero length null components
        # are not supported. So make the null'ed lists empty.
        if validbytes is not None:
            nonzeros = npoffsets[1:] != npoffsets[:-1]
            maskedbytes = validbytes == 0
            if numpy.any(maskedbytes & nonzeros):  # null and count > 0
                new_starts = numpy.asarray(npoffsets[:-1], copy=True)
                new_stops = numpy.asarray(npoffsets[1:], copy=True)
                new_starts[maskedbytes] = 0
                new_stops[maskedbytes] = 0
                next = ak.contents.ListArray(
                    ak.index.Index(new_starts),
                    ak.index.Index(new_stops),
                    self._content,
                    parameters=self._parameters,
                )
                return next.to_ListOffsetArray64(True)._to_arrow(
                    pyarrow, mask_node, validbytes, length, options
                )

        if issubclass(npoffsets.dtype.type, np.int64):
            if downsize and npoffsets[-1] < np.iinfo(np.int32).max:
                npoffsets = numpy.astype(npoffsets, np.int32)

        if issubclass(npoffsets.dtype.type, np.uint32):
            if npoffsets[-1] < np.iinfo(np.int32).max:
                npoffsets = numpy.astype(npoffsets, np.int32)
            else:
                npoffsets = numpy.astype(npoffsets, np.int64)

        if is_string or is_bytestring:
            assert isinstance(akcontent, ak.contents.NumpyArray)

            if issubclass(npoffsets.dtype.type, np.int32):
                if is_string:
                    string_type = pyarrow.string()
                else:
                    string_type = pyarrow.binary()
            else:
                if is_string:
                    string_type = pyarrow.large_string()
                else:
                    string_type = pyarrow.large_binary()

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    string_type,
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                    pyarrow.py_buffer(akcontent._raw(numpy)),
                ],
            )

        else:
            paarray = akcontent._to_arrow(
                pyarrow, None, None, akcontent.length, options
            )

            content_type = pyarrow.list_(paarray.type).value_field.with_nullable(
                akcontent._arrow_needs_option_type()
            )

            if issubclass(npoffsets.dtype.type, np.int32):
                list_type = pyarrow.list_(content_type)
            else:
                list_type = pyarrow.large_list(content_type)

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    list_type,
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(npoffsets),
                ],
                children=[paarray],
                null_count=ak._connect.pyarrow.to_null_count(
                    validbytes, options["count_nulls"]
                ),
            )

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        from packaging.version import parse as parse_version

        cupy = Cupy.instance()
        index = self._offsets.raw(cupy).astype("int32")
        buf = cudf.core.buffer.as_buffer(index)

        if parse_version(cudf.__version__) >= parse_version("24.10.00"):
            ind_buf = cudf.core.column.numerical.NumericalColumn(
                data=buf, dtype=index.dtype, mask=None, size=len(index)
            )
        else:
            ind_buf = cudf.core.column.numerical.NumericalColumn(
                buf, index.dtype, None, size=len(index)
            )
        cont = self._content._to_cudf(cudf, None, len(self._content))
        if mask is not None:
            m = np._module.packbits(mask, bitorder="little")
            if m.nbytes % 64:
                m = cupy.resize(m, ((m.nbytes // 64) + 1) * 64)
            m = cudf.core.buffer.as_buffer(cupy.asarray(m))
        else:
            m = None
        if self.parameters.get("__array__") == "string":
            from cudf.core.column.string import StringColumn

            data = cudf.core.buffer.as_buffer(cupy.asarray(self._content.data))
            # docs for StringColumn says there should be two children instead of a data=
            return StringColumn(
                data=data,
                children=(ind_buf,),
                mask=m,
            )

        if parse_version(cudf.__version__) >= parse_version("24.10.00"):
            return cudf.core.column.lists.ListColumn(
                size=length,
                data=None,
                mask=m,
                children=(ind_buf, cont),
                dtype=cudf.core.dtypes.ListDtype(cont.dtype),
            )
        else:
            return cudf.core.column.lists.ListColumn(
                length,
                mask=m,
                children=(ind_buf, cont),
                dtype=cudf.core.dtypes.ListDtype(cont.dtype),
            )

    def _to_backend_array(self, allow_missing, backend):
        array_param = self.parameter("__array__")
        if array_param == "string":
            # Determine the widest string (in code points)
            _max_code_points = backend.index_nplike.empty(1, dtype=np.int64)
            backend[
                "awkward_NumpyArray_prepare_utf8_to_utf32_padded",
                self._content.dtype.type,
                self._offsets.dtype.type,
                _max_code_points.dtype.type,
            ](
                self._content.data,
                self._offsets.data,
                self._offsets.length,
                _max_code_points,
            )
            max_code_points = backend.index_nplike.index_as_shape_item(
                _max_code_points[0]
            )
            # Ensure that we have at-least length-1 bytestrings
            if max_code_points is not unknown_length:
                max_code_points = max(1, max_code_points)

            # Allocate the correct size buffer
            total_code_points = max_code_points * self.length
            buffer = backend.nplike.empty(total_code_points, dtype=np.uint32)

            # Fill buffer with new uint32_t
            self.backend[
                "awkward_NumpyArray_utf8_to_utf32_padded",
                self._content.dtype.type,
                self._offsets.dtype.type,
                buffer.dtype.type,
            ](
                self._content.data,
                self._offsets.data,
                self._offsets.length,
                max_code_points,
                buffer,
            )
            return buffer.view(np.dtype(("U", max_code_points)))
        elif array_param == "bytestring":
            # Handle length=0 case
            if self.starts.length is not unknown_length and self.starts.length == 0:
                max_count = 0
            else:
                max_count = backend.index_nplike.index_as_shape_item(
                    backend.index_nplike.max(self.stops.data - self.starts.data)
                )

            # Ensure that we have at-least length-1 bytestrings
            if max_count is not unknown_length:
                max_count = max(1, max_count)

            buffer = backend.nplike.empty(max_count * self.length, dtype=np.uint8)

            self.backend[
                "awkward_NumpyArray_pad_zero_to_length",
                self._content.dtype.type,
                self._offsets.dtype.type,
                buffer.dtype.type,
            ](
                self._content.data,
                self._offsets.data,
                self._offsets.length,
                max_count,
                buffer,
            )
            return buffer.view(np.dtype(("S", max_count)))
        else:
            return self.to_RegularArray()._to_backend_array(allow_missing, backend)

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [self]
        else:
            content = self._content[self._offsets[0] : self._offsets[-1]]
            contents = content._remove_structure(backend, options)
            if options["keepdims"]:
                if options["list_to_regular"]:
                    return [
                        ak.contents.RegularArray(
                            c,
                            size=c.length,
                            zeros_length=1,
                            parameters=self._parameters,
                        )
                        for c in contents
                    ]
                else:
                    return [
                        ListOffsetArray(
                            Index64(
                                backend.index_nplike.asarray(
                                    [
                                        0,
                                        backend.index_nplike.shape_item_as_index(
                                            c.length
                                        ),
                                    ]
                                )
                            ),
                            c,
                            parameters=self._parameters,
                        )
                        for c in contents
                    ]
            else:
                return contents

    def _drop_none(self) -> Content:
        if self._content.is_option:
            _, _, none_indexes = self._content._nextcarry_outindex()
            new_content = self._content._drop_none()
            return self._rebuild_without_nones(none_indexes, new_content)
        else:
            return self

    def _rebuild_without_nones(self, none_indexes, new_content):
        new_offsets = Index64.empty(self._offsets.length, self._backend.nplike)

        assert (
            new_offsets.nplike is self._backend.index_nplike
            and self._offsets.nplike is self._backend.index_nplike
            and none_indexes.nplike is self._backend.index_nplike
        )

        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListOffsetArray_drop_none_indexes",
                new_offsets.dtype.type,
                none_indexes.dtype.type,
                self._offsets.dtype.type,
            ](
                new_offsets.data,
                none_indexes.data,
                self._offsets.data,
                self._offsets.length,
                none_indexes.length,
            )
        )
        return ak.contents.ListOffsetArray(new_offsets, new_content)

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if self._backend.nplike.known_data:
            offsetsmin = self._offsets[0]
            offsets = ak.index.Index(
                self._offsets.data - offsetsmin, nplike=self._backend.index_nplike
            )
            content = self._content[offsetsmin : self._offsets[-1]]
        else:
            self._touch_data(recursive=False)
            offsets, content = self._offsets, self._content

        if options["return_array"]:

            def continuation():
                return ListOffsetArray(
                    offsets,
                    content._recursively_apply(
                        action,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    parameters=self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                content._recursively_apply(
                    action,
                    depth + 1,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )

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
        next = self.to_ListOffsetArray64(True)
        next_content = next._content[: next._offsets[-1]]
        return ListOffsetArray(
            next._offsets,
            next_content.to_packed(True) if recursive else next_content,
            parameters=next._parameters,
        )

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        starts, stops = self.starts, self.stops
        starts_data = starts.raw(numpy)
        stops_data = stops.raw(numpy)[: len(starts_data)]

        nonempty = starts_data != stops_data
        if numpy.count_nonzero(nonempty) == 0:
            mini, maxi = 0, 0
        else:
            mini = self._backend.index_nplike.min(starts_data)
            maxi = self._backend.index_nplike.max(stops_data)

        starts_data = starts_data - mini
        stops_data = stops_data - mini

        nextcontent = self._content._getitem_range(mini, maxi)

        if self.parameter("__array__") == "bytestring":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            data = nextcontent.data
            out = [None] * starts.length
            if convert_bytes is None:
                for i in range(starts.length):
                    out[i] = ak._util.tobytes(data[starts_data[i] : stops_data[i]])
            else:
                for i in range(starts.length):
                    out[i] = convert_bytes(
                        ak._util.tobytes(data[starts_data[i] : stops_data[i]])
                    )
            return out

        elif self.parameter("__array__") == "string":
            data = nextcontent.data
            out = [None] * starts.length
            for i in range(starts.length):
                out[i] = ak._util.tobytes(data[starts_data[i] : stops_data[i]]).decode(
                    errors="surrogateescape"
                )
            return out

        else:
            out = self._to_list_custom(behavior, json_conversions)
            if out is not None:
                return out

            content = nextcontent._to_list(behavior, json_conversions)
            out = [None] * starts.length

            for i in range(starts.length):
                out[i] = content[starts_data[i] : stops_data[i]]
            return out

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        offsets = self._offsets.to_nplike(backend.index_nplike)
        return ListOffsetArray(offsets, content, parameters=self._parameters)

    def _awkward_strings_to_nonfinite(self, nonfinit_dict):
        if self.parameter("__array__") == "string":
            strings = self.to_list()
            if any(item in nonfinit_dict for item in strings):
                numbers = self._backend.index_nplike.empty(
                    self.starts.length, dtype=np.float64
                )
                has_another_string = False
                for i, val in enumerate(strings):
                    if val in nonfinit_dict:
                        numbers[i] = nonfinit_dict[val]
                    else:
                        numbers[i] = None
                        has_another_string = True

                content = ak.contents.NumpyArray(numbers)

                if has_another_string:
                    union_tags = ak.index.Index8.zeros(
                        content.length, nplike=self._backend.index_nplike
                    )
                    content.backend.nplike.isnan(content._data, union_tags._data)
                    union_index = Index64(
                        self._backend.index_nplike.arange(
                            content.length, dtype=np.int64
                        ),
                        nplike=self._backend.index_nplike,
                    )

                    return ak.contents.UnionArray(
                        tags=union_tags,
                        index=union_index,
                        contents=[content, self.to_ListOffsetArray64(True)],
                    )

                return content

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and self._offsets.is_equal_to(other.offsets, index_dtype, numpyarray)
            and self._content._is_equal_to(
                other.content, index_dtype, numpyarray, all_parameters
            )
        )
