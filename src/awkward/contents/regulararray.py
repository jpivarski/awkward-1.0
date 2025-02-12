# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.regularmeta import RegularMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import (
    parameters_intersect,
    parameters_union,
    type_parameters_equal,
)
from awkward._regularize import is_integer, is_integer_like
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
from awkward.forms.form import Form, FormKeyPathT
from awkward.forms.regularform import RegularForm
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem
    from awkward.contents.listoffsetarray import ListOffsetArray

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class RegularArray(RegularMeta[Content], Content):
    """
    RegularArray describes lists that all have the same length, the single
    integer `size`. Its underlying `content` is a flattened view of the data;
    that is, each list is not stored separately in memory, but is inferred as a
    subinterval of the underlying data.

    If the `content` length is not an integer multiple of `size`, then the length
    of the RegularArray is truncated to the largest integer multiple.

    An extra field `zeros_length` is ignored unless the `size` is zero. This sets the
    length of the RegularArray in only those cases, so that it is possible for an
    array to contain a non-zero number of zero-length lists with regular type.

    A multidimensional #ak.contents.NumpyArray is equivalent to a one-dimensional
    #ak.layout.NumpyArray nested within several RegularArrays, one for each
    dimension. However, RegularArrays can be used to make lists of any other type.

    Like #ak.contents.ListArray and #ak.contents.ListOffsetArray, a RegularArray can
    represent strings if its `__array__` parameter is `"string"` (UTF-8 assumed) or
    `"bytestring"` (no encoding assumed) and it contains an #ak.contents.NumpyArray
    of `dtype=np.uint8` whose `__array__` parameter is `"char"` (UTF-8 assumed) or
    `"byte"` (no encoding assumed).

    RegularArray corresponds to an Apache Arrow
    [FixedSizeList](https://arrow.apache.org/docs/format/Columnar.html#fixed-size-list-layout).

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class RegularArray(Content):
            def __init__(self, content, size, zeros_length=0):
                assert isinstance(content, Content)
                assert isinstance(size, int)
                assert isinstance(zeros_length, int)
                assert size >= 0
                if size != 0:
                    length = len(content) // size  # floor division
                else:
                    assert zeros_length >= 0
                    length = zeros_length
                self.content = content
                self.size = size
                self.length = length

            def __len__(self):
                return self.length

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    return self.content[(where) * self.size : (where + 1) * self.size]

                elif isinstance(where, slice) and where.step is None:
                    start = where.start * self.size
                    stop = where.stop * self.size
                    zeros_length = where.stop - where.start
                    return RegularArray(
                        self.content[start:stop], self.size, zeros_length
                    )

                elif isinstance(where, str):
                    return RegularArray(self.content[where], self.size, self.length)

                else:
                    raise AssertionError(where)
    """

    def __init__(self, content, size, zeros_length=0, *, parameters=None):
        if not isinstance(content, Content):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Content subtype, not {content!r}"
            )
        if size is unknown_length:
            if content.backend.index_nplike.known_data:
                raise TypeError(
                    f"{type(self).__name__} 'size' must be a non-negative integer for backends with known shapes, not None"
                )
        elif not (is_integer(size) and size >= 0):
            raise TypeError(
                f"{type(self).__name__} 'size' must be a non-negative integer, not {size}"
            )

        if zeros_length is not unknown_length and not (
            is_integer(zeros_length) and zeros_length >= 0
        ):
            raise TypeError(
                f"{type(self).__name__} 'zeros_length' must be a non-negative integer, not {zeros_length}"
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

        self._content = content
        self._size = size
        if content.length is unknown_length or size is unknown_length:
            self._length = unknown_length
        elif size != 0:
            self._length = content.length // size  # floor division
        else:
            self._length = zeros_length
        self._init(parameters, content.backend)

    @property
    def size(self):
        return self._size

    form_cls: Final = RegularForm

    def copy(self, content=UNSET, size=UNSET, zeros_length=UNSET, *, parameters=UNSET):
        return RegularArray(
            self._content if content is UNSET else content,
            self._size if size is UNSET else size,
            self._length if zeros_length is UNSET else zeros_length,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, content, size, zeros_length=0, *, parameters=None):
        return cls(content, size, zeros_length, parameters=parameters)

    @property
    def offsets(self) -> Index:
        return self._compact_offsets64(True)

    @property
    def starts(self) -> Index:
        return self._compact_offsets64(True)[:-1]

    @property
    def stops(self):
        return self._compact_offsets64(True)[1:]

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> RegularForm:
        form_key = getkey(self)
        return self.form_cls(
            self._content._form_with_key(getkey),
            self._size,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> RegularForm:
        return self.form_cls(
            self._content._form_with_key_path((*path, None)),
            self._size,
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
        self._content._to_buffers(form.content, getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        return RegularArray(
            self._content._to_typetracer(forget_length),
            self._size,
            unknown_length if forget_length else self._length,
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<RegularArray size="]
        out.append(repr(str(self._size)))
        out.append(" len=")
        out.append(repr(str(self._length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</RegularArray>")
        out.append(post)
        return "".join(out)

    def to_ListOffsetArray64(self, start_at_zero: bool = False) -> ListOffsetArray:
        offsets = self._compact_offsets64(start_at_zero)
        return self._broadcast_tooffsets64(offsets)

    def to_RegularArray(self):
        return self

    def maybe_to_NumpyArray(self) -> ak.contents.NumpyArray | None:
        content = self._content[: self._length * self._size].maybe_to_NumpyArray()

        if content is not None:
            shape = (self._length, self._size) + content.data.shape[1:]
            return ak.contents.NumpyArray(
                self._backend.nplike.reshape(content.data, shape),
                parameters=parameters_union(self._parameters, content.parameters),
                backend=content.backend,
            )
        else:
            return None

    def _getitem_nothing(self):
        return self._content._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        return False

    def _is_getitem_at_virtual(self) -> bool:
        return False

    def _getitem_at(self, where: IndexType):
        index_nplike = self._backend.index_nplike
        where = index_nplike.regularize_index_for_length(where, self._length)
        size_scalar = index_nplike.shape_item_as_index(self._size)
        start, stop = where * size_scalar, (where + 1) * size_scalar
        return self._content._getitem_range(start, stop)

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        index_nplike = self._backend.index_nplike
        if not index_nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        zeros_length = index_nplike.index_as_shape_item(stop - start)
        substart, substop = (
            start * index_nplike.shape_item_as_index(self._size),
            stop * index_nplike.shape_item_as_index(self._size),
        )
        return RegularArray(
            self._content._getitem_range(substart, substop),
            self._size,
            zeros_length,
            parameters=self._parameters,
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return RegularArray(
            self._content._getitem_field(where, only_fields),
            self._size,
            self._length,
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return RegularArray(
            self._content._getitem_fields(where, only_fields),
            self._size,
            self._length,
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)

        where = carry.data

        copied = allow_lazy == "copied"
        if not issubclass(where.dtype.type, np.int64):
            where = self._backend.index_nplike.astype(where, dtype=np.int64)
            copied = True

        negative = where < 0
        if self._backend.index_nplike.known_data:
            if self._backend.index_nplike.any(negative):
                if not copied:
                    where = where.copy()
                    copied = True
                where[negative] += self._length

            if self._backend.index_nplike.any(where >= self._length):
                raise ak._errors.index_error(self, where)

        if where.shape[0] is unknown_length or self._size is unknown_length:
            nextcarry = ak.index.Index64.empty(
                unknown_length, self._backend.index_nplike
            )
        else:
            nextcarry = ak.index.Index64.empty(
                where.shape[0] * self._size, self._backend.index_nplike
            )
        assert nextcarry.nplike is self._backend.index_nplike
        self._maybe_index_error(
            self._backend[
                "awkward_RegularArray_getitem_carry",
                nextcarry.dtype.type,
                where.dtype.type,
            ](
                nextcarry.data,
                where,
                where.shape[0],
                self._size,
            ),
            slicer=carry,
        )

        return RegularArray(
            self._content._carry(nextcarry, allow_lazy),
            self._size,
            where.shape[0],
            parameters=self._parameters,
        )

    def _compact_offsets64(self, start_at_zero):
        index_nplike = self._backend.index_nplike
        if self._size is not unknown_length and self._size == 0:
            return ak.index.Index64.zeros(self._length + 1, nplike=index_nplike)
        else:
            return ak.index.Index64(
                index_nplike.arange(
                    0,
                    index_nplike.shape_item_as_index(self._length * self._size) + 1,
                    index_nplike.shape_item_as_index(self._size),
                    dtype=np.int64,
                ),
                nplike=index_nplike,
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
            and self._length is not unknown_length
            and offsets.length - 1 != self._length
        ):
            raise AssertionError(
                f"cannot broadcast RegularArray of length {self._length} to length {offsets.length - 1}"
            )

        if self._size is not unknown_length and self._size == 1:
            count = offsets.data[1:] - offsets.data[:-1]
            # Sanity check that our kernel isn't losing values here
            assert (
                not self._backend.index_nplike.known_data
                or count.size is unknown_length
                or count.size == 0
                or count.dtype == np.intp
                or self._backend.index_nplike.max(count) <= np.iinfo(np.intp).max
            )
            carry = ak.index.Index64(
                index_nplike.repeat(
                    index_nplike.arange(
                        index_nplike.shape_item_as_index(self._length), dtype=np.int64
                    ),
                    index_nplike.astype(count, np.intp),
                ),
                nplike=index_nplike,
            )
            next_content = self._content._carry(carry, True)
        else:
            this_offsets = self._compact_offsets64(True)
            if index_nplike.known_data and not index_nplike.array_equal(
                offsets.data, this_offsets.data
            ):
                raise ValueError("cannot broadcast nested list")

            next_content = self._content[: offsets[-1]]
        return ak.contents.ListOffsetArray(
            offsets, next_content, parameters=self._parameters
        )

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        out = self.to_ListOffsetArray64(True)
        return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

    def _getitem_next(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        index_nplike = self._backend.index_nplike

        if head is NO_HEAD:
            return self

        elif is_integer_like(head):
            nexthead, nexttail = ak._slicing.head_tail(tail)
            nextcarry = ak.index.Index64.empty(self._length, index_nplike)
            assert nextcarry.nplike is index_nplike
            head = ak._slicing.normalize_integer_like(head)
            self._maybe_index_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_at", nextcarry.dtype.type
                ](
                    nextcarry.data,
                    head,
                    self._length,
                    self._size,
                ),
                slicer=head,
            )
            nextcontent = self._content._carry(nextcarry, True)
            return nextcontent._getitem_next(nexthead, nexttail, advanced)

        elif isinstance(head, slice):
            nexthead, nexttail = ak._slicing.head_tail(tail)
            start, stop, step, nextsize = index_nplike.derive_slice_for_length(
                head, length=self._size
            )

            nextcarry = ak.index.Index64.empty(self._length * nextsize, index_nplike)
            assert nextcarry.nplike is index_nplike
            self._maybe_index_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_range",
                    nextcarry.dtype.type,
                ](
                    nextcarry.data,
                    start,
                    step,
                    self._length,
                    self._size,
                    nextsize,
                ),
                slicer=head,
            )

            nextcontent = self._content._carry(nextcarry, True)

            if advanced is None or (
                advanced.length is not unknown_length and advanced.length == 0
            ):
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, advanced),
                    nextsize,
                    self._length,
                    parameters=self._parameters,
                )
            else:
                nextadvanced = ak.index.Index64.empty(nextcarry.length, index_nplike)
                advanced = advanced.to_nplike(index_nplike)
                assert (
                    nextadvanced.nplike is index_nplike
                    and advanced.nplike is index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_range_spreadadvanced",
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                    ](
                        nextadvanced.data,
                        advanced.data,
                        self._length,
                        nextsize,
                    ),
                    slicer=head,
                )
                return RegularArray(
                    nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
                    nextsize,
                    self._length,
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
            head = head.to_nplike(index_nplike)
            flathead = index_nplike.reshape(index_nplike.asarray(head.data), (-1,))
            regular_flathead = ak.index.Index64.empty(flathead.shape[0], index_nplike)
            assert regular_flathead.nplike is index_nplike
            self._maybe_index_error(
                self._backend[
                    "awkward_RegularArray_getitem_next_array_regularize",
                    regular_flathead.dtype.type,
                    flathead.dtype.type,
                ](
                    regular_flathead.data,
                    flathead,
                    flathead.shape[0],
                    self._size,
                ),
                slicer=head,
            )

            nexthead, nexttail = ak._slicing.head_tail(tail)
            if advanced is None or (
                advanced.length is not unknown_length and advanced.length == 0
            ):
                nextcarry = ak.index.Index64.empty(
                    self._length * flathead.shape[0],
                    index_nplike,
                )
                nextadvanced = ak.index.Index64.empty(nextcarry.length, index_nplike)
                assert (
                    nextcarry.nplike is index_nplike
                    and nextadvanced.nplike is index_nplike
                    and regular_flathead.nplike is index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_array",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        regular_flathead.data,
                        self._length,
                        regular_flathead.length,
                        self._size,
                    ),
                    slicer=head,
                )
                nextcontent = self._content._carry(nextcarry, True)

                out = nextcontent._getitem_next(nexthead, nexttail, nextadvanced)
                if advanced is None:
                    return ak._slicing.getitem_next_array_wrap(
                        out, head.metadata.get("shape", (head.length,)), self._length
                    )
                else:
                    return out

            elif self._size == 0:
                nextcarry = ak.index.Index64.empty(0, nplike=index_nplike)
                nextadvanced = ak.index.Index64.empty(0, nplike=index_nplike)
                nextcontent = self._content._carry(nextcarry, True)
                return nextcontent._getitem_next(nexthead, nexttail, nextadvanced)

            else:
                nextcarry = ak.index.Index64.empty(self._length, index_nplike)
                nextadvanced = ak.index.Index64.empty(self._length, index_nplike)
                advanced = advanced.to_nplike(index_nplike)
                assert (
                    nextcarry.nplike is index_nplike
                    and nextadvanced.nplike is index_nplike
                    and advanced.nplike is index_nplike
                    and regular_flathead.nplike is index_nplike
                )
                self._maybe_index_error(
                    self._backend[
                        "awkward_RegularArray_getitem_next_array_advanced",
                        nextcarry.dtype.type,
                        nextadvanced.dtype.type,
                        advanced.dtype.type,
                        regular_flathead.dtype.type,
                    ](
                        nextcarry.data,
                        nextadvanced.data,
                        advanced.data,
                        regular_flathead.data,
                        self._length,
                        self._size,
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

            if self._backend.nplike.known_data and head.length != self._size:
                raise ak._errors.index_error(
                    self,
                    head,
                    f"cannot fit jagged slice with length {head.length} into {type(self).__name__} of size {self._size}",
                )

            multistarts = ak.index.Index64.empty(
                head.length * self._length, index_nplike
            )
            multistops = ak.index.Index64.empty(
                head.length * self._length, index_nplike
            )

            assert head.offsets.nplike is index_nplike
            self._maybe_index_error(
                self._backend[
                    "awkward_RegularArray_getitem_jagged_expand",
                    multistarts.dtype.type,
                    multistops.dtype.type,
                    head.offsets.dtype.type,
                ](
                    multistarts.data,
                    multistops.data,
                    head.offsets.data,
                    head.length,
                    self._length,
                ),
                slicer=head,
            )
            down = self._content._getitem_next_jagged(
                multistarts, multistops, head._content, tail
            )

            return RegularArray(
                down, headlength, self._length, parameters=self._parameters
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

        if any(x.is_option for x in others):
            return ak.contents.UnmaskedArray(self)._mergemany(others)

        # Regularize NumpyArray into RegularArray (or NumpyArray if 1D)
        others = [
            o.to_RegularArray() if isinstance(o, ak.contents.NumpyArray) else o
            for o in others
        ]

        if all(x.is_regular and x.size == self.size for x in others):
            parameters = self._parameters
            tail_contents = []
            zeros_length = self._length
            for x in others:
                parameters = parameters_intersect(parameters, x._parameters)
                tail_contents.append(x._content[: x._length * x._size])
                zeros_length += x._length

            return RegularArray(
                self._content[: self._length * self._size]._mergemany(tail_contents),
                self._size,
                zeros_length,
                parameters=parameters,
            )

        else:
            return self.to_ListOffsetArray64(True)._mergemany(others)

    def _fill_none(self, value: Content) -> Content:
        return RegularArray(
            self._content._fill_none(value),
            self._size,
            self._length,
            parameters=self._parameters,
        )

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        elif posaxis is not None and posaxis + 1 == depth + 1:
            localindex = ak.index.Index64.empty(
                self._length * self._size, nplike=self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend["awkward_RegularArray_localindex", np.int64](
                    localindex.data,
                    self._size,
                    self._length,
                )
            )
            return ak.contents.RegularArray(
                ak.contents.NumpyArray(localindex.data), self._size, self._length
            )
        else:
            return ak.contents.RegularArray(
                self._content._local_index(axis, depth + 1), self._size, self._length
            )

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.RegularArray(
            self._content._numbers_to_type(name, including_unknown),
            self._size,
            self._length,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._length == 0:
            return True

        return self.to_ListOffsetArray64(True)._is_unique(
            negaxis,
            starts,
            parents,
            outlength,
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._length == 0:
            return self
        out = self.to_ListOffsetArray64(True)._unique(
            negaxis,
            starts,
            parents,
            outlength,
        )

        if isinstance(out, ak.contents.RegularArray):
            if isinstance(out._content, ak.contents.ListOffsetArray):
                return ak.contents.RegularArray(
                    out._content.to_RegularArray(),
                    out._size,
                    out._length,
                    parameters=out._parameters,
                )

        return out

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        next = self.to_ListOffsetArray64(True)
        out = next._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

        if isinstance(out, ak.contents.RegularArray):
            if isinstance(out._content, ak.contents.ListOffsetArray):
                return ak.contents.RegularArray(
                    out._content.to_RegularArray(),
                    out._size,
                    out._length,
                    parameters=out._parameters,
                )

        return out

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        out = self.to_ListOffsetArray64(True)._sort_next(
            negaxis, starts, parents, outlength, ascending, stable
        )

        # FIXME
        # if isinstance(out, ak.contents.RegularArray):
        #     if isinstance(out._content, ak.contents.ListOffsetArray):
        #         return ak.contents.RegularArray(
        #             out._content.to_RegularArray(),
        #             out._size,
        #             out._length,
        #             None,
        #             out._parameters,
        #             self._backend.nplike,
        #         )

        return out

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

            size = self._size
            if replacement:
                size = size + (n - 1)
            thisn = n
            if thisn > size:
                combinationslen = 0
            elif thisn == size:
                combinationslen = 1
            else:
                if thisn * 2 > size:
                    thisn = size - thisn
                combinationslen = size
                for j in range(2, thisn + 1):
                    combinationslen = combinationslen * (size - j + 1)
                    combinationslen = combinationslen // j

            totallen = combinationslen * self._length
            tocarryraw = index_nplike.empty(n, dtype=np.intp)
            tocarry = []
            for i in range(n):
                ptr = ak.index.Index64.empty(
                    totallen,
                    nplike=index_nplike,
                    dtype=np.int64,
                )
                tocarry.append(ptr)
                if self._backend.nplike.known_data:
                    tocarryraw[i] = ptr.ptr

            toindex = ak.index.Index64.empty(n, index_nplike, dtype=np.int64)
            fromindex = ak.index.Index64.empty(n, index_nplike, dtype=np.int64)

            if self._size != 0:
                assert (
                    toindex.nplike is index_nplike and fromindex.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_RegularArray_combinations_64",
                        np.int64,
                        toindex.data.dtype.type,
                        fromindex.data.dtype.type,
                    ](
                        tocarryraw,
                        toindex.data,
                        fromindex.data,
                        n,
                        replacement,
                        self._size,
                        self._length,
                    )
                )

            contents = []
            length = UNSET
            for ptr in tocarry:
                contents.append(self._content._carry(ptr, True))
                length = contents[-1].length
            assert length is not UNSET
            recordarray = ak.contents.RecordArray(
                contents,
                recordlookup,
                length,
                parameters=parameters,
                backend=self._backend,
            )
            return ak.contents.RegularArray(
                recordarray, combinationslen, self._length, parameters=self._parameters
            )
        else:
            length = self._length * self._size
            next = self._content._getitem_range(
                0, index_nplike.shape_item_as_index(length)
            )._combinations(n, replacement, recordlookup, parameters, axis, depth + 1)
            return ak.contents.RegularArray(
                next, self._size, self._length, parameters=self._parameters
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
        branch, depth = self.branch_depth
        nextlen = self._length * self._size
        if not branch and negaxis == depth:
            nextcarry = ak.index.Index64.empty(nextlen, nplike=index_nplike)
            nextparents = ak.index.Index64.empty(nextlen, nplike=index_nplike)
            assert (
                parents.nplike is index_nplike
                and nextcarry.nplike is index_nplike
                and nextparents.nplike is index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_RegularArray_reduce_nonlocal_preparenext_64",
                    nextcarry.dtype.type,
                    nextparents.dtype.type,
                    parents.dtype.type,
                ](
                    nextcarry.data,
                    nextparents.data,
                    parents.data,
                    self._size,
                    self._length,
                )
            )
            nextstarts = ak.index.Index64.empty(
                # `starts` must have at least enough elements for the largest `nextparent` to index into
                # The upper bound for this value is given by `nextlen` (each item in this list belonging
                # to a distinct reduction), but the length of `starts` should equate to `maxnextparents - 1`.
                starts.length * self._size,
                nplike=index_nplike,
            )
            assert (
                nextstarts.nplike is index_nplike and nextparents.nplike is index_nplike
            )
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

            if reducer.needs_position:
                # Regular arrays have the same length rows, so there can be no "missing" values
                # unlike ragged list types
                nextshifts = ak.index.Index64.zeros(
                    nextcarry.length, nplike=index_nplike
                )
            else:
                nextshifts = None

            nextcontent = self._content._carry(nextcarry, False)
            nextoutlength = outlength * self._size
            outcontent = nextcontent._reduce_next(
                reducer,
                negaxis - 1,
                nextstarts,
                nextshifts,
                nextparents,
                # We want a result of length
                nextoutlength,
                mask,
                False,
                behavior,
            )

            out = ak.contents.RegularArray(
                outcontent, self._size, outlength, parameters=None
            )

            if keepdims:
                out = ak.contents.RegularArray(out, 1, self._length, parameters=None)
            return out
        else:
            nextparents = ak.index.Index64.empty(nextlen, index_nplike)

            assert nextparents.nplike is index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_RegularArray_reduce_local_nextparents_64",
                    nextparents.dtype.type,
                ](
                    nextparents.data,
                    self._size,
                    self._length,
                )
            )

            if self._size is not unknown_length and self._size > 0:
                nextstarts = ak.index.Index64(
                    index_nplike.arange(0, nextlen, self._size),
                    nplike=index_nplike,
                )
            else:
                assert nextlen is unknown_length or nextlen == 0
                nextstarts = ak.index.Index64(
                    index_nplike.zeros(nextlen, dtype=np.int64),
                    nplike=index_nplike,
                )

            outcontent = self._content._reduce_next(
                reducer,
                negaxis,
                nextstarts,
                shifts,
                nextparents,
                self._length,
                mask,
                keepdims,
                behavior,
            )

            # Understanding this logic is nontrivial without examples. So, let's start with a
            # somewhat simple example of `5 * 4 * 3 * int64`. This is composed of the following layouts
            #                                     ^........^ NumpyArray(5*4*3)
            #                                 ^............^ RegularArray(..., size=3)
            #                             ^................^ RegularArray(..., size=4)
            # If the reduction axis is 2 (or -1), then the resulting shape is `5 * 4 * int64`.
            # This corresponds to `RegularArray(NumpyArray(5*4), size=4)`, i.e. the `size=3` layout
            # *disappears*, and is replaced by the bare reduction `NumpyArray`. Therefore, if
            # this layout is _directly_ above the reduction, it won't survive the reduction.
            #
            # The `_reduce_next` mechanism returns its first result (by recursion) from the
            # leaf `NumpyArray`. The parent `RegularArray` takes the `negaxis != depth` branch,
            # and asks the `NumpyArray` to perform the reduction. The `_reduce_next` is such that the
            # callee must wrap the reduction result into a list whose length is given by the *caller*,
            # i.e. the parent. Therefore, the `RegularArray(..., size=3)` layout reduces its content with
            # `outlength=len(self)=5*4`. The parent of this `RegularArray` (with `size=4`)
            # will reduce its contents with `outlength=5`. It is *this* reduction that must be returned as a
            # `RegularArray(..., size=4)`. Clearly, the `RegularArray(..., size=3)` layout has disappeared
            # and been replaced by a `NumpyArray` of the appropriate size for the `RegularArray(..., size=4)`
            # to wrap. Hence, we only want to interpret the content as a `RegularArray(..., size=self._size)`
            # iff. we are not the direct parent of the reduced layout.

            # At `depth == negaxis+1`, we are above the dimension being reduced. All implemented
            # _dimensional_ types should return `RegularArray` for `keepdims=True`, so we don't need
            # to convert `outcontent` to a `RegularArray`
            if keepdims and depth == negaxis + 1:
                assert outcontent.is_regular
            # At `depth >= negaxis + 2`, we are wrapping at _least_ one other list type. This list-type
            # may return a ListOffsetArray, which we need to convert to a `RegularArray`. We know that
            # the result can be reinterpreted as a `RegularArray`, because it's not the immediate parent
            # of the reduction, so it must exist in the type.
            elif depth >= negaxis + 2:
                if outcontent.is_list:
                    # Let's only deal with ListOffsetArray
                    outcontent = outcontent.to_ListOffsetArray64(False)
                    # Fast-path to convert data that we know should be regular (avoid a kernel call)
                    start, stop = (
                        outcontent.offsets[0],
                        outcontent.offsets[outcontent.offsets.length - 1],
                    )

                    trimmed = outcontent.content._getitem_range(start, stop)
                    assert (
                        trimmed.length is unknown_length
                        or outcontent.length is unknown_length
                        or trimmed.length == self._size * outcontent.length
                    )

                    outcontent = ak.contents.RegularArray(
                        trimmed,
                        size=self._size,
                        zeros_length=self._length,
                    )
                else:
                    assert outcontent.is_regular

            outoffsets = ak.index.Index64.empty(outlength + 1, index_nplike)
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

            return ak.contents.ListOffsetArray(outoffsets, outcontent, parameters=None)

    def _validity_error(self, path):
        if self.size < 0:
            return f"at {path} ({type(self)!r}): size < 0"

        return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)

        elif posaxis is not None and posaxis + 1 == depth + 1:
            if not clip:
                if target < self._size:
                    return self
                else:
                    return self._pad_none(target, axis, depth, True)

            else:
                index = ak.index.Index64.empty(
                    self._length * target,
                    self._backend.index_nplike,
                )
                assert index.nplike is self._backend.index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_RegularArray_rpad_and_clip_axis1", index.dtype.type
                    ](index.data, target, self._size, self._length)
                )
                next = ak.contents.IndexedOptionArray.simplified(
                    index, self._content, parameters=self._parameters
                )
                return ak.contents.RegularArray(
                    next,
                    target,
                    self._length,
                    parameters=self._parameters,
                )

        else:
            return ak.contents.RegularArray(
                self._content._pad_none(target, axis, depth + 1, clip),
                self._size,
                self._length,
                parameters=self._parameters,
            )

    def _to_backend_array(self, allow_missing, backend):
        array_param = self.parameter("__array__")
        if array_param == "string":
            offsets = self._compact_offsets64(True)
            # Determine the widest string (in code points)
            _max_code_points = backend.index_nplike.empty(1, dtype=np.int64)
            backend[
                "awkward_NumpyArray_prepare_utf8_to_utf32_padded",
                self._content.dtype.type,
                offsets.dtype.type,
                _max_code_points.dtype.type,
            ](
                self._content.data,
                offsets.data,
                offsets.length,
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
                offsets.dtype.type,
                buffer.dtype.type,
            ](
                self._content.data,
                offsets.data,
                offsets.length,
                max_code_points,
                buffer,
            )
            return buffer.view(np.dtype(("U", max_code_points)))
        elif array_param == "bytestring":
            # Ensure that we have at-least length-1 bytestrings
            if self._size is not unknown_length and self._size == 0:
                # Create new empty-buffer
                return backend.nplike.zeros(self.length, dtype=np.uint8).view(
                    np.dtype(("S", 1))
                )
            else:
                return self._content.data.view(np.dtype(("S", self._size)))
        else:
            out = self._content._to_backend_array(allow_missing, backend)
            shape = (self._length, self._size) + out.shape[1:]

            # ShapeItem is a defined type, but some nplikes don't map onto the entire space; e.g.
            # NumPy never has `None` shape items. We require that if a shape-item is used between nplikes
            # they both be the same "known-shape-ness".
            assert (
                self._backend.index_nplike.known_data == self._backend.nplike.known_data
            )
            return self._backend.nplike.reshape(
                out[
                    : self._backend.nplike.shape_item_as_index(
                        self._length * self._size
                    )
                ],
                shape,
            )

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        assert self._backend.nplike.known_data

        if self.parameter("__array__") == "string":
            return self.to_ListOffsetArray64(False)._to_arrow(
                pyarrow, mask_node, validbytes, length, options
            )

        is_bytestring = self.parameter("__array__") == "bytestring"

        akcontent = self._content[: self._length * self._size]

        if is_bytestring:
            assert isinstance(akcontent, ak.contents.NumpyArray)

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.binary(self._size),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                self._length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                    pyarrow.py_buffer(akcontent._raw(numpy)),
                ],
            )

        else:
            paarray = akcontent._to_arrow(
                pyarrow, None, None, self._length * self._size, options
            )

            content_type = pyarrow.list_(paarray.type).value_field.with_nullable(
                akcontent._arrow_needs_option_type()
            )

            return pyarrow.Array.from_buffers(
                ak._connect.pyarrow.to_awkwardarrow_type(
                    pyarrow.list_(content_type, self._size),
                    options["extensionarray"],
                    options["record_is_scalar"],
                    mask_node,
                    self,
                ),
                self._length,
                [
                    ak._connect.pyarrow.to_validbits(validbytes),
                ],
                children=[paarray],
                null_count=ak._connect.pyarrow.to_null_count(
                    validbytes, options["count_nulls"]
                ),
            )

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        if (
            self.parameter("__array__") == "string"
            or self.parameter("__array__") == "bytestring"
        ):
            return [self]
        else:
            index_nplike = self._backend.index_nplike
            content = self._content[
                : index_nplike.shape_item_as_index(self._length * self._size)
            ]
            contents = content._remove_structure(backend, options)
            if options["keepdims"]:
                return [
                    RegularArray(c, size=c.length, parameters=self._parameters)
                    for c in contents
                ]
            else:
                return contents

    def _drop_none(self) -> Content:
        return self.to_ListOffsetArray64()._drop_none()

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if options["regular_to_jagged"]:
            return self.to_ListOffsetArray64(False)._recursively_apply(
                action, depth, depth_context, lateral_context, options
            )

        if self._backend.nplike.known_data:
            content = self._content[: self._length * self._size]
        else:
            self._touch_data(recursive=False)
            content = self._content

        if options["return_array"]:

            def continuation():
                return RegularArray(
                    content._recursively_apply(
                        action,
                        depth + 1,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._size,
                    self._length,
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
        index_nplike = self._backend.index_nplike
        length = self._length * self._size
        content = self._content[: index_nplike.shape_item_as_index(length)]

        return RegularArray(
            content.to_packed(True) if recursive else content,
            self._size,
            self._length,
            parameters=self._parameters,
        )

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        if self.parameter("__array__") == "bytestring":
            convert_bytes = (
                None if json_conversions is None else json_conversions["convert_bytes"]
            )
            content = ak._util.tobytes(self._content.data)
            length, size = self._length, self._size
            out = [None] * length
            if convert_bytes is None:
                for i in range(length):
                    out[i] = content[(i) * size : (i + 1) * size]
            else:
                for i in range(length):
                    out[i] = convert_bytes(content[(i) * size : (i + 1) * size])
            return out

        elif self.parameter("__array__") == "string":
            data = self._content.data
            if hasattr(data, "tobytes"):

                def tostring(x):
                    return x.tobytes().decode(errors="surrogateescape")

            else:

                def tostring(x):
                    return x.tostring().decode(errors="surrogateescape")

            length, size = self._length, self._size
            out = [None] * length
            for i in range(length):
                out[i] = tostring(data[(i) * size : (i + 1) * size])
            return out

        else:
            out = self._to_list_custom(behavior, json_conversions)
            if out is not None:
                return out

            content = self._content._to_list(behavior, json_conversions)
            length, size = self._length, self._size
            out = [None] * length
            for i in range(length):
                out[i] = content[(i) * size : (i + 1) * size]
            return out

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        return RegularArray(
            content, self._size, zeros_length=self._length, parameters=self._parameters
        )

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and (
                self._size is unknown_length
                or other.size is unknown_length
                or self._size == other.size
            )
            and self._content._is_equal_to(
                other._content, index_dtype, numpyarray, all_parameters
            )
        )
