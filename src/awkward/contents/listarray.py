# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.listmeta import ListMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import TypeTracer
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
from awkward.contents.listoffsetarray import ListOffsetArray
from awkward.forms.form import Form, FormKeyPathT
from awkward.forms.listform import ListForm
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()


@final
class ListArray(ListMeta[Content], Content):
    """
    ListArray generalizes #ak.contents.ListOffsetArray by not
    requiring its `content` to be in increasing order and by allowing it to
    have unreachable elements between lists. Instead of a single `offsets` buffer,
    ListArray has

    * `starts`: The starting index of each list.
    * `stops`: The stopping index of each list.

    #ak.contents.ListOffsetArray `offsets` may be related to `starts` and
    `stops` by

        starts = offsets[:-1]
        stops = offsets[1:]

    ListArrays are a common by-product of structure manipulation: as a result of
    some operation, we might want to view slices or permutations of the `content`
    without copying it to make a contiguous version of it. For that reason,
    ListArrays are more useful in a data-manipulation library like Awkward Array
    than in a data-representation library like Apache Arrow.

    Like #ak.contents.ListOffsetArray and #ak.contents.RegularArray, a ListArray can
    represent strings if its `__array__` parameter is `"string"` (UTF-8 assumed) or
    `"bytestring"` (no encoding assumed) and it contains an #ak.contents.NumpyArray
    of `dtype=np.uint8` whose `__array__` parameter is `"char"` (UTF-8 assumed) or
    `"byte"` (no encoding assumed).

    There is no equivalent of ListArray in Apache Arrow.

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class ListArray(Content):
            def __init__(self, starts, stops, content):
                assert isinstance(starts, (Index32, IndexU32, Index64))
                assert isinstance(stops, type(starts))
                assert isinstance(content, Content)
                assert len(stops) >= len(starts)  # usually equal
                for i in range(len(starts)):
                    start = starts[i]
                    stop = stops[i]
                    if start != stop:
                        assert start < stop  # i.e. start <= stop
                        assert start >= 0
                        assert stop <= len(content)
                self.starts = starts
                self.stops = stops
                self.content = content

            def __len__(self):
                return len(self.starts)

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    return self.content[self.starts[where] : self.stops[where]]

                elif isinstance(where, slice) and where.step is None:
                    starts = self.starts[where.start : where.stop]
                    stops = self.stops[where.start : where.stop]
                    return ListArray(starts, stops, self.content)

                elif isinstance(where, str):
                    return ListArray(self.starts, self.stops, self.content[where])

                else:
                    raise AssertionError(where)
    """

    def __init__(self, starts, stops, content, *, parameters=None):
        if not isinstance(starts, Index) and starts.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                f"{type(self).__name__} 'starts' must be an Index with dtype in (int32, uint32, int64), "
                f"not {starts!r}"
            )
        if not (isinstance(stops, Index) and starts.dtype == stops.dtype):
            raise TypeError(
                f"{type(self).__name__} 'stops' must be an Index with the same dtype as 'starts' ({starts.dtype!r}), "
                f"not {stops!r}"
            )
        if not isinstance(content, Content):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Content subtype, not {content!r}"
            )
        if content.backend.index_nplike.known_data and starts.length > stops.length:
            raise ValueError(
                f"{type(self).__name__} len(starts) ({starts.length}) must be <= len(stops) ({stops.length})"
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

        assert starts.nplike is content.backend.index_nplike
        assert stops.nplike is content.backend.index_nplike

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(parameters, content.backend)

    @property
    def starts(self) -> Index:
        return self._starts

    @property
    def stops(self) -> Index:
        return self._stops

    form_cls: Final = ListForm

    def copy(self, starts=UNSET, stops=UNSET, content=UNSET, *, parameters=UNSET):
        return ListArray(
            self._starts if starts is UNSET else starts,
            self._stops if stops is UNSET else stops,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            starts=copy.deepcopy(self._starts, memo),
            stops=copy.deepcopy(self._stops, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, starts, stops, content, *, parameters=None):
        return cls(starts, stops, content, parameters=parameters)

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> ListForm:
        form_key = getkey(self)
        return self.form_cls(
            self._starts.form,
            self._stops.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> ListForm:
        return self.form_cls(
            self._starts.form,
            self._stops.form,
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
        key1 = getkey(self, form, "starts")
        key2 = getkey(self, form, "stops")
        container[key1] = ak._util.native_to_byteorder(
            self._starts.raw(backend.index_nplike), byteorder
        )
        container[key2] = ak._util.native_to_byteorder(
            self._stops.raw(backend.index_nplike), byteorder
        )
        self._content._to_buffers(form.content, getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        tt = TypeTracer.instance()
        starts = self._starts.to_nplike(tt)
        return ListArray(
            starts.forget_length() if forget_length else starts,
            self._stops.to_nplike(tt),
            self._content._to_typetracer(forget_length),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        self._starts._touch_data()
        self._stops._touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        self._starts._touch_shape()
        self._stops._touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._starts.length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ListArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._starts._repr(indent + "    ", "<starts>", "</starts>\n"))
        out.append(self._stops._repr(indent + "    ", "<stops>", "</stops>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ListArray>")
        out.append(post)
        return "".join(out)

    def to_ListOffsetArray64(self, start_at_zero: bool = False) -> ListOffsetArray:
        index_nplike = self._backend.index_nplike

        starts = self._starts.data
        stops = self._stops.data

        lenoffsets = self._starts.length + 1
        if (not index_nplike.known_data) or index_nplike.array_equal(
            starts[1:], stops[:-1]
        ):
            offsets = index_nplike.empty(lenoffsets, dtype=starts.dtype)
            if lenoffsets is not unknown_length and lenoffsets == 1:
                offsets[0] = 0
            else:
                offsets[:-1] = starts
                offsets[-1] = stops[-1]
            return ListOffsetArray(
                ak.index.Index(offsets, nplike=self._backend.index_nplike),
                self._content,
                parameters=self._parameters,
            ).to_ListOffsetArray64(start_at_zero=start_at_zero)

        else:
            offsets = self._compact_offsets64(start_at_zero)
            return self._broadcast_tooffsets64(offsets)

    def to_RegularArray(self):
        offsets = self._compact_offsets64(True)
        return self._broadcast_tooffsets64(offsets).to_RegularArray()

    def _getitem_nothing(self):
        return self._content._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder_starts = isinstance(self._starts.data, PlaceholderArray)
        is_placeholder_stops = isinstance(self._stops.data, PlaceholderArray)
        is_placeholder = is_placeholder_starts or is_placeholder_stops
        return is_placeholder

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual_starts = (
            isinstance(self._starts.data, VirtualArray)
            and not self._starts.data.is_materialized
        )
        is_virtual_stops = (
            isinstance(self._stops.data, VirtualArray)
            and not self._stops.data.is_materialized
        )
        is_virtual = is_virtual_starts or is_virtual_stops
        return is_virtual

    def _getitem_at(self, where: IndexType):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return self._content._getitem_range(0, 0)

        if where < 0:
            where += self.length
        if not (0 <= where < self.length) and self._backend.nplike.known_data:
            raise ak._errors.index_error(self, where)
        start, stop = self._starts[where], self._stops[where]
        return self._content._getitem_range(start, stop)

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        return ListArray(
            self._starts[start:stop],
            self._stops[start:stop],
            self._content,
            parameters=self._parameters,
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ListArray(
            self._starts,
            self._stops,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)

        try:
            nextstarts = self._starts[carry.data]
            nextstops = self._stops[: self._starts.length][carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return ListArray(
            nextstarts, nextstops, self._content, parameters=self._parameters
        )

    def _compact_offsets64(self, start_at_zero):
        starts_len = self._starts.length
        out = ak.index.Index64.empty(
            starts_len + 1,
            self._backend.index_nplike,
        )
        assert (
            out.nplike is self._backend.index_nplike
            and self._starts.nplike is self._backend.index_nplike
            and self._stops.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListArray_compact_offsets",
                out.dtype.type,
                self._starts.dtype.type,
                self._stops.dtype.type,
            ](
                out.data,
                self._starts.data,
                self._stops.data,
                starts_len,
            )
        )
        return out

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
            and self._starts.length is not unknown_length
            and offsets.length - 1 != self._starts.length
        ):
            raise AssertionError(
                f"cannot broadcast RegularArray of length {self._starts.length} to length {offsets.length - 1}"
            )

        nextcarry = ak.index.Index64.empty(
            self._backend.index_nplike.index_as_shape_item(offsets[-1]),
            self._backend.index_nplike,
        )
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and offsets.nplike is self._backend.index_nplike
            and self._starts.nplike is self._backend.index_nplike
            and self._stops.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ListArray_broadcast_tooffsets",
                nextcarry.dtype.type,
                offsets.dtype.type,
                self._starts.dtype.type,
                self._stops.dtype.type,
            ](
                nextcarry.data,
                offsets.data,
                offsets.length,
                self._starts.data,
                self._stops.data,
                self._content.length,
            )
        )

        nextcontent = self._content._carry(nextcarry, True)

        return ListOffsetArray(offsets, nextcontent, parameters=self._parameters)

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        slicestarts = slicestarts.to_nplike(self._backend.index_nplike)
        slicestops = slicestops.to_nplike(self._backend.index_nplike)
        if self._backend.nplike.known_data and slicestarts.length != self.length:
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                f"cannot fit jagged slice with length {slicestarts.length} into {type(self).__name__} of size {self.length}",
            )

        if isinstance(slicecontent, ak.contents.ListOffsetArray):
            outoffsets = ak.index.Index64.empty(
                slicestarts.length + 1,
                self._backend.index_nplike,
            )
            assert (
                outoffsets.nplike is self._backend.index_nplike
                and slicestarts.nplike is self._backend.index_nplike
                and slicestops.nplike is self._backend.index_nplike
                and self._starts.nplike is self._backend.index_nplike
                and self._stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_descend",
                    outoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    self._starts.data,
                    self._stops.data,
                ),
                slicer=ListArray(slicestarts, slicestops, slicecontent),
            )

            as_list_offset_array = self.to_ListOffsetArray64(False)
            next_content = as_list_offset_array._content[
                as_list_offset_array.offsets[0] : as_list_offset_array.offsets[-1]
            ]

            sliceoffsets = slicecontent._offsets

            outcontent = next_content._getitem_next_jagged(
                sliceoffsets[:-1], sliceoffsets[1:], slicecontent._content, tail
            )

            return ak.contents.ListOffsetArray(
                outoffsets, outcontent, parameters=self._parameters
            )

        elif isinstance(slicecontent, ak.contents.NumpyArray):
            _carrylen = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                _carrylen.nplike is self._backend.index_nplike
                and slicestarts.nplike is self._backend.index_nplike
                and slicestops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_carrylen",
                    _carrylen.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                ](
                    _carrylen.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                ),
                slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
            )
            carrylen = self._backend.index_nplike.index_as_shape_item(_carrylen[0])

            sliceindex = ak.index.Index64(slicecontent._data)
            outoffsets = ak.index.Index64.empty(
                slicestarts.length + 1,
                self._backend.index_nplike,
            )
            nextcarry = ak.index.Index64.empty(carrylen, self._backend.index_nplike)

            assert (
                outoffsets.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and slicestarts.nplike is self._backend.index_nplike
                and slicestops.nplike is self._backend.index_nplike
                and sliceindex.nplike is self._backend.index_nplike
                and self._starts.nplike is self._backend.index_nplike
                and self._stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_apply",
                    outoffsets.dtype.type,
                    nextcarry.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    sliceindex.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    outoffsets.data,
                    nextcarry.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    sliceindex.data,
                    sliceindex.length,
                    self._starts.data,
                    self._stops.data,
                    self._content.length,
                ),
                slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
            )
            nextcontent = self._content._carry(nextcarry, True)
            nexthead, nexttail = ak._slicing.head_tail(tail)
            outcontent = nextcontent._getitem_next(nexthead, nexttail, None)

            return ak.contents.ListOffsetArray(outoffsets, outcontent, parameters=None)

        elif isinstance(slicecontent, ak.contents.IndexedOptionArray):
            if (
                self._backend.nplike.known_data
                and self._starts.length < slicestarts.length
            ):
                raise ak._errors.index_error(
                    self,
                    ak.contents.ListArray(
                        slicestarts, slicestops, slicecontent, parameters=None
                    ),
                    "jagged slice length differs from array length",
                )

            missing = slicecontent._index
            _numvalid = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                _numvalid.nplike is self._backend.index_nplike
                and slicestarts.nplike is self._backend.index_nplike
                and slicestops.nplike is self._backend.index_nplike
                and missing.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_numvalid",
                    _numvalid.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    _numvalid.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    missing.data,
                    missing.length,
                ),
                slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
            )
            numvalid = self._backend.index_nplike.index_as_shape_item(_numvalid[0])

            nextcarry = ak.index.Index64.empty(numvalid, self._backend.index_nplike)

            smalloffsets = ak.index.Index64.empty(
                slicestarts.length + 1,
                self._backend.index_nplike,
            )
            largeoffsets = ak.index.Index64.empty(
                slicestarts.length + 1,
                self._backend.index_nplike,
            )

            assert (
                nextcarry.nplike is self._backend.index_nplike
                and smalloffsets.nplike is self._backend.index_nplike
                and largeoffsets.nplike is self._backend.index_nplike
                and slicestarts.nplike is self._backend.index_nplike
                and slicestops.nplike is self._backend.index_nplike
                and missing.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_shrink",
                    nextcarry.dtype.type,
                    smalloffsets.dtype.type,
                    largeoffsets.dtype.type,
                    slicestarts.dtype.type,
                    slicestops.dtype.type,
                    missing.dtype.type,
                ](
                    nextcarry.data,
                    smalloffsets.data,
                    largeoffsets.data,
                    slicestarts.data,
                    slicestops.data,
                    slicestarts.length,
                    missing.data,
                ),
                slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
            )

            if isinstance(
                slicecontent._content,
                ak.contents.ListOffsetArray,
            ):
                # Generate ranges between starts and stops
                as_list_offset_array = self.to_ListOffsetArray64(True)
                nextcontent = as_list_offset_array._content._carry(nextcarry, True)
                next = ak.contents.ListOffsetArray(
                    smalloffsets, nextcontent, parameters=self._parameters
                )
                out = next._getitem_next_jagged(
                    smalloffsets[:-1], smalloffsets[1:], slicecontent._content, tail
                )

            else:
                out = self._getitem_next_jagged(
                    smalloffsets[:-1], smalloffsets[1:], slicecontent._content, tail
                )

            if isinstance(out, ak.contents.ListOffsetArray):
                content = out._content
                if largeoffsets.nplike.known_data:
                    missing_trim = missing[0 : largeoffsets[-1]]
                else:
                    missing_trim = missing
                out = ak.contents.IndexedOptionArray.simplified(
                    missing_trim, content, parameters=self._parameters
                )
                return ak.contents.ListOffsetArray(
                    largeoffsets,
                    out,
                    parameters=self._parameters,
                )
            else:
                raise AssertionError(
                    f"expected ListOffsetArray from ListArray._getitem_next_jagged, got {type(out).__name__}"
                )

        elif isinstance(slicecontent, ak.contents.EmptyArray):
            return self

        else:
            raise AssertionError(
                f"expected Index/IndexedOptionArray/ListOffsetArray in ListArray._getitem_next_jagged, got {type(slicecontent).__name__}"
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
            assert advanced is None
            nexthead, nexttail = ak._slicing.head_tail(tail)
            lenstarts = self._starts.length
            nextcarry = ak.index.Index64.empty(lenstarts, self._backend.index_nplike)
            head = ak._slicing.normalize_integer_like(head)
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._starts.nplike is self._backend.index_nplike
                and self._stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_next_at",
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
                    lenstarts,
                    head,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            lenstarts = self._starts.length

            nexthead, nexttail = ak._slicing.head_tail(tail)

            start, stop, step = head.start, head.stop, head.step

            step = 1 if step is None else step
            start = ak._util.kSliceNone if start is None else start
            stop = ak._util.kSliceNone if stop is None else stop

            if self._backend.nplike.known_data:
                carrylength = ak.index.Index64.empty(1, self._backend.index_nplike)
                assert (
                    carrylength.nplike is self._backend.index_nplike
                    and self._starts.nplike is self._backend.index_nplike
                    and self._stops.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_range_carrylength",
                        carrylength.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        carrylength.data,
                        self._starts.data,
                        self._stops.data,
                        lenstarts,
                        start,
                        stop,
                        step,
                    ),
                    slicer=head,
                )
                nextcarry = ak.index.Index64.empty(
                    carrylength[0], self._backend.index_nplike
                )
            else:
                self._touch_data(recursive=False)
                nextcarry = ak.index.Index64.empty(
                    unknown_length, self._backend.index_nplike
                )

            lennextoffsets = lenstarts + 1
            if self._starts.dtype == "int64":
                nextoffsets = ak.index.Index64.empty(
                    lennextoffsets, self._backend.index_nplike
                )
            elif self._starts.dtype == "int32":
                nextoffsets = ak.index.Index32.empty(
                    lennextoffsets, self._backend.index_nplike
                )
            elif self._starts.dtype == "uint32":
                nextoffsets = ak.index.IndexU32.empty(
                    lennextoffsets, self._backend.index_nplike
                )

            assert (
                nextoffsets.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and self._starts.nplike is self._backend.index_nplike
                and self._stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_next_range",
                    nextoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    nextoffsets.data,
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
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
                if self._backend.nplike.known_data:
                    total = ak.index.Index64.empty(1, self._backend.index_nplike)
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
                    nextadvanced = ak.index.Index64.empty(
                        total[0], self._backend.index_nplike
                    )
                else:
                    self._touch_data(recursive=False)
                    nextadvanced = ak.index.Index64.empty(
                        unknown_length, self._backend.index_nplike
                    )
                advanced = advanced.to_nplike(self._backend.index_nplike)
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

        elif isinstance(head, ak.index.Index64):
            lenstarts = self._starts.length

            nexthead, nexttail = ak._slicing.head_tail(tail)
            flathead = self._backend.index_nplike.reshape(
                self._backend.index_nplike.asarray(head.data), (-1,)
            )
            regular_flathead = ak.index.Index64(
                flathead, nplike=self._backend.index_nplike
            )
            if advanced is None or (
                advanced.length is not unknown_length and advanced.length == 0
            ):
                nextcarry = ak.index.Index64.empty(
                    lenstarts * flathead.shape[0],
                    self._backend.index_nplike,
                )
                nextadvanced = ak.index.Index64.empty(
                    lenstarts * flathead.shape[0],
                    self._backend.index_nplike,
                )
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and self._starts.nplike is self._backend.index_nplike
                    and self._stops.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self._starts.data,
                        self._stops.data,
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
                        out, head.metadata.get("shape", (head.length,)), self.length
                    )
                else:
                    return out

            else:
                nextcarry = ak.index.Index64.empty(
                    lenstarts, self._backend.index_nplike
                )
                nextadvanced = ak.index.Index64.empty(
                    lenstarts, self._backend.index_nplike
                )
                advanced = advanced.to_nplike(self._backend.index_nplike)
                assert (
                    nextcarry.nplike is self._backend.index_nplike
                    and nextadvanced.nplike is self._backend.index_nplike
                    and self._starts.nplike is self._backend.index_nplike
                    and self._stops.nplike is self._backend.index_nplike
                    and regular_flathead.nplike is self._backend.index_nplike
                    and advanced.nplike is self._backend.index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_ListArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                        regular_flathead.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        self._starts.data,
                        self._stops.data,
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
            headlength = head.length
            head = head.to_backend(self._backend)
            if advanced is not None:
                raise ak._errors.index_error(
                    self,
                    head,
                    "cannot mix jagged slice with NumPy-style advanced indexing",
                )
            length = self.length
            singleoffsets = ak.index.Index64(head.offsets.data)
            multistarts = ak.index.Index64.empty(
                head.length * length, self._backend.index_nplike
            )
            multistops = ak.index.Index64.empty(
                head.length * length, self._backend.index_nplike
            )
            nextcarry = ak.index.Index64.empty(
                head.length * length, self._backend.index_nplike
            )

            assert (
                multistarts.nplike is self._backend.index_nplike
                and multistops.nplike is self._backend.index_nplike
                and singleoffsets.nplike is self._backend.index_nplike
                and nextcarry.nplike is self._backend.index_nplike
                and self._starts.nplike is self._backend.index_nplike
                and self._stops.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_ListArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    singleoffsets.dtype.type,
                    nextcarry.dtype.type,
                    self._starts.dtype.type,
                    self._stops.dtype.type,
                ](
                    multistarts.data,
                    multistops.data,
                    singleoffsets.data,
                    nextcarry.data,
                    self._starts.data,
                    self._stops.data,
                    head.length,
                    length,
                ),
                slicer=head,
            )
            carried = self._content._carry(nextcarry, True)
            down = carried._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return ak.contents.RegularArray(
                down, headlength, 1, parameters=self._parameters
            )

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        return self.to_ListOffsetArray64(True)._offsets_and_flattened(axis, depth)

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

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        contents = []

        parameters = self._parameters

        for array in head:
            if isinstance(array, ak.contents.EmptyArray):
                continue

            parameters = parameters_intersect(parameters, array._parameters)
            if isinstance(
                array,
                (
                    ak.contents.ListArray,
                    ak.contents.ListOffsetArray,
                    ak.contents.RegularArray,
                ),
            ):
                contents.append(array.content)
            elif array.is_numpy:
                contents.append(array.to_RegularArray().content)
            else:
                raise ValueError(
                    "cannot merge "
                    + type(self).__name__
                    + " with "
                    + type(array).__name__
                    + "."
                )

        tail_contents = contents[1:]
        nextcontent = contents[0]._mergemany(tail_contents)

        nextstarts = ak.index.Index64.empty(total_length, self._backend.index_nplike)
        nextstops = ak.index.Index64.empty(total_length, self._backend.index_nplike)

        contentlength_so_far = 0
        length_so_far = 0

        for array in head:
            # We need contiguous content, so let's just convert to RegularArray
            # immediately.
            if array.is_numpy:
                array = array.to_RegularArray()

            if isinstance(
                array,
                (
                    ak.contents.ListArray,
                    ak.contents.ListOffsetArray,
                ),
            ):
                array_starts = array.starts
                array_stops = array.stops

                assert (
                    nextstarts.nplike is self._backend.index_nplike
                    and nextstops.nplike is self._backend.index_nplike
                    and array_starts.nplike is self._backend.index_nplike
                    and array_stops.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.data,
                        length_so_far,
                        nextstops.data,
                        length_so_far,
                        array_starts.data,
                        array_stops.data,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length

                length_so_far += array.length

            elif isinstance(array, ak.contents.RegularArray):
                listoffsetarray = array.to_ListOffsetArray64(True)

                array_starts = listoffsetarray.starts
                array_stops = listoffsetarray.stops

                assert (
                    nextstarts.nplike is self._backend.index_nplike
                    and nextstops.nplike is self._backend.index_nplike
                    and array_starts.nplike is self._backend.index_nplike
                    and array_stops.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListArray_fill",
                        nextstarts.dtype.type,
                        nextstops.dtype.type,
                        array_starts.dtype.type,
                        array_stops.dtype.type,
                    ](
                        nextstarts.data,
                        length_so_far,
                        nextstops.data,
                        length_so_far,
                        array_starts.data,
                        array_stops.data,
                        listoffsetarray.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length

                length_so_far += array.length

            elif isinstance(array, ak.contents.EmptyArray):
                pass

            else:
                raise AssertionError

        next = ak.contents.ListArray(
            nextstarts, nextstops, nextcontent, parameters=parameters
        )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        return ListArray(
            self._starts,
            self._stops,
            self._content._fill_none(value),
            parameters=self._parameters,
        )

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif posaxis is not None and posaxis + 1 == depth + 1:
            offsets = self._compact_offsets64(True)
            innerlength = self._backend.index_nplike.index_as_shape_item(
                offsets[-1]
            )  # todo: removed touch_data?
            localindex = ak.index.Index64.empty(innerlength, self._backend.index_nplike)
            assert (
                localindex.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
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
            return ak.contents.ListArray(
                self._starts,
                self._stops,
                self._content._local_index(axis, depth + 1),
            )

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.ListArray(
            self._starts,
            self._stops,
            self._content._numbers_to_type(name, including_unknown),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._starts.length is not unknown_length and self._starts.length == 0:
            return True

        return self.to_ListOffsetArray64(True)._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._starts.length is not unknown_length and self._starts.length == 0:
            return self

        return self.to_ListOffsetArray64(True)._unique(
            negaxis, starts, parents, outlength
        )

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        next = self.to_ListOffsetArray64(True)
        out = next._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )
        return out

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        return self.to_ListOffsetArray64(True)._sort_next(
            negaxis, starts, parents, outlength, ascending, stable
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return ListOffsetArray._combinations(
            self, n, replacement, recordlookup, parameters, axis, depth
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
        return self.to_ListOffsetArray64(True)._reduce_next(
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
        if self._backend.nplike.known_data and self.stops.length < self.starts.length:
            return f"at {path} ({type(self)!r}): len(stops) < len(starts)"
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
        return (
            self.starts._nbytes_part()
            + self.stops._nbytes_part()
            + self.content._nbytes_part()
        )

    def _pad_none(self, target, axis, depth, clip):
        if not clip:
            posaxis = maybe_posaxis(self, axis, depth)
            if posaxis is not None and posaxis + 1 == depth:
                return self._pad_none_axis0(target, clip)
            elif posaxis is not None and posaxis + 1 == depth + 1:
                _min = ak.index.Index64.empty(1, self._backend.index_nplike)
                assert (
                    _min.nplike is self._backend.index_nplike
                    and self._starts.nplike is self._backend.index_nplike
                    and self._stops.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ListArray_min_range",
                        _min.dtype.type,
                        self._starts.dtype.type,
                        self._stops.dtype.type,
                    ](
                        _min.data,
                        self._starts.data,
                        self._stops.data,
                        self._starts.length,
                    )
                )
                min_ = self._backend.index_nplike.index_as_shape_item(_min[0])
                # TODO: Replace the kernel call with below code once typtracer supports '-'
                # min_ = self._backend.nplike.min(self._stops.data - self._starts.data)
                if min_ is not unknown_length and target < min_:
                    return self
                else:
                    _tolength = ak.index.Index64.empty(1, self._backend.index_nplike)
                    assert (
                        _tolength.nplike is self._backend.index_nplike
                        and self._starts.nplike is self._backend.index_nplike
                        and self._stops.nplike is self._backend.index_nplike
                    )
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_ListArray_rpad_and_clip_length_axis1",
                            _tolength.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                        ](
                            _tolength.data,
                            self._starts.data,
                            self._stops.data,
                            target,
                            self._starts.length,
                        )
                    )
                    tolength = self._backend.index_nplike.index_as_shape_item(
                        _tolength[0]
                    )

                    index = ak.index.Index64.empty(tolength, self._backend.index_nplike)
                    starts_ = ak.index.Index.empty(
                        self._starts.length,
                        self._backend.index_nplike,
                        dtype=self._starts.dtype,
                    )
                    stops_ = ak.index.Index.empty(
                        self._stops.length,
                        self._backend.index_nplike,
                        dtype=self._stops.dtype,
                    )
                    assert (
                        index.nplike is self._backend.index_nplike
                        and self._starts.nplike is self._backend.index_nplike
                        and self._stops.nplike is self._backend.index_nplike
                        and starts_.nplike is self._backend.index_nplike
                        and stops_.nplike is self._backend.index_nplike
                    )
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_ListArray_rpad_axis1",
                            index.dtype.type,
                            self._starts.dtype.type,
                            self._stops.dtype.type,
                            starts_.dtype.type,
                            stops_.dtype.type,
                        ](
                            index.data,
                            self._starts.data,
                            self._stops.data,
                            starts_.data,
                            stops_.data,
                            target,
                            self._starts.length,
                        )
                    )
                    next = ak.contents.IndexedOptionArray.simplified(
                        index, self._content, parameters=None
                    )
                    return ak.contents.ListArray(
                        starts_,
                        stops_,
                        next,
                        parameters=self._parameters,
                    )
            else:
                return ak.contents.ListArray(
                    self._starts,
                    self._stops,
                    self._content._pad_none(target, axis, depth + 1, clip),
                    parameters=self._parameters,
                )
        else:
            return self.to_ListOffsetArray64(True)._pad_none(
                target, axis, depth, clip=True
            )

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        return self.to_ListOffsetArray64(False)._to_arrow(
            pyarrow, mask_node, validbytes, length, options
        )

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        return self.to_ListOffsetArray64(False)._to_cudf(cudf, mask, length)

    def _to_backend_array(self, allow_missing, backend):
        array_param = self.parameter("__array__")
        if array_param in {"bytestring", "string"}:
            return self.to_ListOffsetArray64(False)._to_backend_array(
                allow_missing, backend
            )
        else:
            return self.to_RegularArray()._to_backend_array(allow_missing, backend)

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        return self.to_ListOffsetArray64(False)._remove_structure(backend, options)

    def _drop_none(self) -> Content:
        return self.to_ListOffsetArray64()._drop_none()

    def _rebuild_without_nones(self, none_indexes, new_content):
        return self.to_ListOffsetArray64()._rebuild_without_nones(
            none_indexes, new_content
        )

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if (
            self._backend.nplike.known_data
            and self._backend.nplike.known_data
            and self._starts.length != 0
        ):
            startsmin = self._backend.index_nplike.min(self._starts.data)
            starts = ak.index.Index(
                self._starts.data - startsmin, nplike=self._backend.index_nplike
            )
            stops = ak.index.Index(
                self._stops.data - startsmin, nplike=self._backend.index_nplike
            )
            content = self._content[
                startsmin : self._backend.index_nplike.max(self._stops.data)
            ]
        else:
            self._touch_data(recursive=False)
            starts, stops, content = self._starts, self._stops, self._content

        if options["return_array"]:

            def continuation():
                return ListArray(
                    starts,
                    stops,
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
        return self.to_ListOffsetArray64(True).to_packed(recursive)

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        return ListOffsetArray._to_list(self, behavior, json_conversions)

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        starts = self._starts.to_nplike(backend.index_nplike)
        stops = self._stops.to_nplike(backend.index_nplike)
        return ListArray(starts, stops, content, parameters=self._parameters)

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and self._starts.is_equal_to(other.starts, index_dtype, numpyarray)
            and self._stops.is_equal_to(other.stops, index_dtype, numpyarray)
            and self._content._is_equal_to(
                other.content, index_dtype, numpyarray, all_parameters
            )
        )
