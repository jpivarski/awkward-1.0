# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.indexedoptionmeta import IndexedOptionMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import MaybeNone, TypeTracer
from awkward._nplikes.virtual import VirtualArray
from awkward._parameters import (
    parameters_intersect,
    parameters_union,
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
from awkward.forms.indexedoptionform import IndexedOptionForm
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class IndexedOptionArray(IndexedOptionMeta[Content], Content):
    """
    IndexedOptionArray is an #ak.contents.IndexedArray for which
    negative values in the index are interpreted as missing. It represents
    #ak.types.OptionType data like #ak.contents.ByteMaskedArray,
    #ak.contents.BitMaskedArray, and #ak.contents.UnmaskedArray, but
    the flexibility of the arbitrary `index` makes it a common output of
    many operations.

    IndexedOptionArray doesn't have a direct equivalent in Apache Arrow.

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class IndexedOptionArray(Content):
            def __init__(self, index, content):
                assert isinstance(index, (Index32, Index64))
                assert isinstance(content, Content)
                for x in index:
                    assert x < len(content)  # index[i] may be negative
                self.index = index
                self.content = content

            def __len__(self):
                return len(self.index)

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    if self.index[where] < 0:
                        return None
                    else:
                        return self.content[self.index[where]]

                elif isinstance(where, slice) and where.step is None:
                    return IndexedOptionArray(
                        self.index[where.start : where.stop], self.content
                    )

                elif isinstance(where, str):
                    return IndexedOptionArray(self.index, self.content[where])

                else:
                    raise AssertionError(where)
    """

    def __init__(self, index, content, *, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.int64),
            )
        ):
            raise TypeError(
                f"{type(self).__name__} 'index' must be an Index with dtype in (int32, uint32, int64), "
                f"not {index!r}"
            )
        if not isinstance(content, Content):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Content subtype, not {content!r}"
            )
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"
        if (content.is_union and not is_cat) or content.is_indexed or content.is_option:
            raise TypeError(
                "{0} cannot contain a union-type (unless categorical), option-type, or indexed 'content' ({1}); try {0}.simplified instead".format(
                    type(self).__name__, type(content).__name__
                )
            )

        assert index.nplike is content.backend.index_nplike

        self._index = index
        self._content = content
        self._init(parameters, content.backend)

    @property
    def index(self):
        return self._index

    form_cls: Final = IndexedOptionForm

    def copy(self, index=UNSET, content=UNSET, *, parameters=UNSET):
        return IndexedOptionArray(
            self._index if index is UNSET else index,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            index=copy.deepcopy(self._index, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(cls, index, content, *, parameters=None):
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"

        if content.is_union and not is_cat:
            return content._union_of_optionarrays(index, parameters)

        elif content.is_indexed or content.is_option:
            backend = content.backend
            if content.is_indexed:
                inner = content.index
            else:
                inner = content.to_IndexedOptionArray64().index
            result = ak.index.Index64.empty(index.length, nplike=backend.index_nplike)

            backend.maybe_kernel_error(
                backend[
                    "awkward_IndexedArray_simplify",
                    result.dtype.type,
                    index.dtype.type,
                    inner.dtype.type,
                ](
                    result.data,
                    index.data,
                    index.length,
                    inner.data,
                    inner.length,
                )
            )
            return IndexedOptionArray(
                result,
                content.content,
                parameters=parameters_union(content._parameters, parameters),
            )

        else:
            return cls(index, content, parameters=parameters)

    def _form_with_key(
        self, getkey: Callable[[Content], str | None]
    ) -> IndexedOptionForm:
        form_key = getkey(self)
        return self.form_cls(
            self._index.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> IndexedOptionForm:
        return self.form_cls(
            self._index.form,
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
        key = getkey(self, form, "index")
        container[key] = ak._util.native_to_byteorder(
            self._index.raw(backend.index_nplike), byteorder
        )
        self._content._to_buffers(form.content, getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        index = self._index.to_nplike(TypeTracer.instance())
        return IndexedOptionArray(
            index.forget_length() if forget_length else index,
            self._content._to_typetracer(forget_length),
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        self._index._touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        self._index._touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._index.length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<IndexedOptionArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedOptionArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self) -> IndexedOptionArray:
        if self._index.dtype == np.dtype(np.int64):
            return self
        else:
            return IndexedOptionArray(
                self._backend.index_nplike.astype(self._index, dtype=np.int64),
                self._content,
                parameters=self._parameters,
            )

    def to_ByteMaskedArray(self, valid_when):
        mask = ak.index.Index8(self.mask_as_bool(valid_when))

        carry = self._index.data
        too_negative = carry < -1
        if self._backend.index_nplike.known_data and self._backend.index_nplike.any(
            too_negative
        ):
            carry = carry.copy()
            carry[too_negative] = -1
        carry = ak.index.Index(carry)

        if self._content.length is not unknown_length and self._content.length == 0:
            content = self._content.form.length_one_array(backend=self._backend)._carry(
                carry, False
            )
        else:
            content = self._content._carry(carry, False)

        return ak.contents.ByteMaskedArray(
            mask, content, valid_when, parameters=self._parameters
        )

    def to_BitMaskedArray(self, valid_when, lsb_order):
        return self.to_ByteMaskedArray(valid_when).to_BitMaskedArray(
            valid_when, lsb_order
        )

    def mask_as_bool(self, valid_when: bool = True) -> ArrayLike:
        if valid_when:
            return self._index.raw(self._backend.index_nplike) >= 0
        else:
            return self._index.raw(self._backend.index_nplike) < 0

    def _getitem_nothing(self):
        return self._content._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder = isinstance(self._index.data, PlaceholderArray)
        if is_placeholder:
            return True
        return self._content._is_getitem_at_placeholder()

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual = (
            isinstance(self._index.data, VirtualArray)
            and not self._index.data.is_materialized
        )
        if is_virtual:
            return True
        return self._content._is_getitem_at_virtual()

    def _getitem_at(self, where: IndexType):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return MaybeNone(self._content._getitem_at(where))

        if where < 0:
            where += self.length
        if self._backend.nplike.known_data and not 0 <= where < self.length:
            raise ak._errors.index_error(self, where)
        if self._index[where] < 0:
            return None
        else:
            return self._content._getitem_at(self._index[where])

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        return IndexedOptionArray(
            self._index[start:stop], self._content, parameters=self._parameters
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return IndexedOptionArray.simplified(
            self._index,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return IndexedOptionArray.simplified(
            self._index,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> IndexedOptionArray:
        assert isinstance(carry, ak.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return IndexedOptionArray(nextindex, self._content, parameters=self._parameters)

    def _nextcarry_outindex(self):
        backend = self._backend
        index_nplike = backend.index_nplike

        _numnull = ak.index.Index64.empty(1, nplike=backend.index_nplike)
        assert _numnull.nplike is index_nplike and self._index.nplike is index_nplike
        self._backend.maybe_kernel_error(
            backend[
                "awkward_IndexedArray_numnull",
                _numnull.dtype.type,
                self._index.dtype.type,
            ](
                _numnull.data,
                self._index.data,
                self._index.length,
            )
        )
        numnull = index_nplike.index_as_shape_item(_numnull[0])
        nextcarry = ak.index.Index64.empty(
            self._index.length - numnull,
            index_nplike,
        )
        outindex = ak.index.Index.empty(
            self._index.length,
            index_nplike,
            dtype=self._index.dtype,
        )

        assert (
            nextcarry.nplike is index_nplike
            and outindex.nplike is index_nplike
            and self._index.nplike is index_nplike
        )
        self._backend.maybe_kernel_error(
            backend[
                "awkward_IndexedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.data,
                outindex.data,
                self._index.data,
                self._index.length,
                self._content.length,
            )
        )

        return numnull, nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
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

        numnull, nextcarry, outindex = self._nextcarry_outindex()

        reducedstarts = ak.index.Index64.empty(
            self.length - numnull,
            nplike=self._backend.index_nplike,
        )
        reducedstops = ak.index.Index64.empty(
            self.length - numnull,
            nplike=self._backend.index_nplike,
        )
        assert (
            outindex.nplike is self._backend.index_nplike
            and slicestarts.nplike is self._backend.index_nplike
            and slicestops.nplike is self._backend.index_nplike
            and reducedstarts.nplike is self._backend.index_nplike
            and reducedstops.nplike is self._backend.index_nplike
        )
        self._maybe_index_error(
            self._backend[
                "awkward_MaskedArray_getitem_next_jagged_project",
                outindex.dtype.type,
                slicestarts.dtype.type,
                slicestops.dtype.type,
                reducedstarts.dtype.type,
                reducedstops.dtype.type,
            ](
                outindex.data,
                slicestarts.data,
                slicestops.data,
                reducedstarts.data,
                reducedstops.data,
                self.length,
            ),
            slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
        )
        next = self._content._carry(nextcarry, True)
        out = next._getitem_next_jagged(reducedstarts, reducedstops, slicecontent, tail)

        return ak.contents.IndexedOptionArray.simplified(
            outindex, out, parameters=self._parameters
        )

    def _getitem_next_jagged(
        self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    ) -> Content:
        return self._getitem_next_jagged_generic(
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

        elif is_integer_like(head) or isinstance(
            head, (slice, ak.index.Index64, ak.contents.ListOffsetArray)
        ):
            nexthead, nexttail = ak._slicing.head_tail(tail)

            numnull, nextcarry, outindex = self._nextcarry_outindex()

            next = self._content._carry(nextcarry, True)
            out = next._getitem_next(head, tail, advanced)
            return IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list) and isinstance(head[0], str):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self, mask=None):
        if mask is not None:
            if self._backend.nplike.known_data and self._index.length != mask.length:
                raise ValueError(
                    f"mask length ({mask.length()}) is not equal to {type(self).__name__} length ({self._index.length})"
                )
            nextindex = ak.index.Index64.empty(
                self._index.length, self._backend.index_nplike
            )
            assert (
                nextindex.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_overlay_mask",
                    nextindex.dtype.type,
                    mask.dtype.type,
                    self._index.dtype.type,
                ](
                    nextindex.data,
                    mask.data,
                    self._index.data,
                    self._index.length,
                )
            )
            next = ak.contents.IndexedOptionArray(
                nextindex, self._content, parameters=self._parameters
            )
            return next.project()
        else:
            _numnull = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                _numnull.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_numnull",
                    _numnull.dtype.type,
                    self._index.dtype.type,
                ](
                    _numnull.data,
                    self._index.data,
                    self._index.length,
                )
            )
            numnull = self._backend.index_nplike.index_as_shape_item(_numnull[0])
            nextcarry = ak.index.Index64.empty(
                self.length - numnull,
                self._backend.index_nplike,
            )

            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_flatten_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.data,
                    self._index.data,
                    self._index.length,
                    self._content.length,
                )
            )

            return self._content._carry(nextcarry, True)

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError("axis=0 not allowed for flatten")
        else:
            numnull, nextcarry, outindex = self._nextcarry_outindex()
            next = self._content._carry(nextcarry, False)

            offsets, flattened = next._offsets_and_flattened(axis, depth)

            if offsets.length is not unknown_length and offsets.length == 0:
                return (
                    offsets,
                    ak.contents.IndexedOptionArray(
                        outindex, flattened, parameters=self._parameters
                    ),
                )

            else:
                outoffsets = ak.index.Index64.empty(
                    offsets.length + numnull,
                    self._backend.index_nplike,
                    dtype=np.int64,
                )

                assert (
                    outoffsets.nplike is self._backend.index_nplike
                    and outindex.nplike is self._backend.index_nplike
                    and offsets.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_IndexedArray_flatten_none2empty",
                        outoffsets.dtype.type,
                        outindex.dtype.type,
                        offsets.dtype.type,
                    ](
                        outoffsets.data,
                        outindex.data,
                        outindex.length,
                        offsets.data,
                        offsets.length,
                    )
                )
                return (outoffsets, flattened)

    def _mergeable_next(self, other: Content, mergebool: bool) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # Is the other array indexed or optional?
        elif other.is_option or other.is_indexed:
            return self._content._mergeable_next(other.content, mergebool)
        else:
            return self._content._mergeable_next(other, mergebool)

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ValueError(
                "to merge this array with 'others', at least one other must be provided"
            )

        head = [self]
        tail = []

        it_others = iter(others)
        for other in it_others:
            if isinstance(other, ak.contents.UnionArray):
                tail.append(other)
                tail.extend(it_others)
                break
            else:
                head.append(other)

        if any(x.backend.nplike.known_data for x in head + tail) and not all(
            x.backend.nplike.known_data for x in head + tail
        ):
            raise RuntimeError

        return head, tail

    def _reverse_merge(self, other):
        if isinstance(other, ak.contents.EmptyArray):
            return self

        # FIXME: support categorical-categorical merging
        if (
            other.is_indexed
            and other.parameter("__array__")
            == self.parameter("__array__")
            == "categorical"
        ):
            raise NotImplementedError(
                "merging categorical arrays is currently not implemented. "
                "Use `ak.enforce_type` to drop the categorical type and use general merging."
            )

        theirlength = other.length
        mylength = self.length
        index = ak.index.Index64.empty(
            theirlength + mylength,
            self._backend.index_nplike,
        )

        content = other._mergemany([self._content])

        # Fill index::0→theirlength with arange(theirlength)
        assert index.nplike is self._backend.index_nplike
        self._backend.maybe_kernel_error(
            self._backend["awkward_IndexedArray_fill_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )

        # Fill index::theirlength->end with self.index[:mylength]+theirlength
        assert (
            index.nplike is self._backend.index_nplike
            and self.index.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_fill",
                index.dtype.type,
                self.index.dtype.type,
            ](
                index.data,
                theirlength,
                self.index.data,
                mylength,
                theirlength,
            )
        )
        # We can directly merge with other options and indexed types, but we must merge parameters
        if other.is_option or other.is_indexed:
            parameters = parameters_union(self._parameters, other._parameters)
        # Otherwise, this option parameters win out
        else:
            parameters = self._parameters

        return ak.contents.IndexedOptionArray.simplified(
            index, content, parameters=parameters
        )

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self

        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        contents = []
        contentlength_so_far = 0
        length_so_far = 0
        nextindex = ak.index.Index64.empty(total_length, self._backend.index_nplike)
        parameters = self._parameters

        for array in head:
            if isinstance(array, ak.contents.EmptyArray):
                continue

            if isinstance(
                array,
                (
                    ak.contents.ByteMaskedArray,
                    ak.contents.BitMaskedArray,
                    ak.contents.UnmaskedArray,
                ),
            ):
                array = array.to_IndexedOptionArray64()

            if isinstance(
                array, (ak.contents.IndexedOptionArray, ak.contents.IndexedArray)
            ):
                array = array._trim()  # see: #3185 and #3119
                # If we're merging an option, then merge parameters before pulling out `content`
                parameters = parameters_intersect(parameters, array._parameters)
                contents.append(array.content)
                array_index = array.index
                assert (
                    nextindex.nplike is self._backend.index_nplike
                    and array_index.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_IndexedArray_fill",
                        nextindex.dtype.type,
                        array_index.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        array_index.data,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.content.length

                length_so_far += array.length

            else:
                contents.append(array)
                assert nextindex.nplike is self._backend.index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_IndexedArray_fill_count",
                        nextindex.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        array.length,
                        contentlength_so_far,
                    )
                )
                contentlength_so_far += array.length

                length_so_far += array.length

                # Categoricals may only survive if all contents are categorical
                if (
                    parameters is not None
                    and parameters.get("__array__") == "categorical"
                ):
                    parameters = {**parameters}
                    del parameters["__array__"]

        tail_contents = contents[1:]
        nextcontent = contents[0]._mergemany(tail_contents)
        next = ak.contents.IndexedOptionArray(
            nextindex, nextcontent, parameters=parameters
        )

        # FIXME: support categorical merging?
        if parameters is not None and parameters.get("__array__") == "categorical":
            raise NotImplementedError(
                "merging categorical arrays is currently not implemented. "
                "Use `ak.enforce_type` to drop the categorical type and use general merging."
            )

        if len(tail) == 0:
            return next

        reversed = tail[0]._reverse_merge(next)
        if len(tail) == 1:
            return reversed
        else:
            return reversed._mergemany(tail[1:])

    def _fill_none(self, value: Content) -> Content:
        if value.backend.nplike.known_data and value.length != 1:
            raise ValueError(
                f"fill_none value length ({value.length}) is not equal to 1"
            )

        contents = [self._content, value]
        tags = ak.index.Index8(self.mask_as_bool(valid_when=False))
        index = ak.index.Index64.empty(tags.length, self._backend.index_nplike)

        assert (
            index.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_UnionArray_fillna", index.dtype.type, self._index.dtype.type
            ](index.data, self._index.data, tags.length)
        )
        return ak.contents.UnionArray.simplified(
            tags,
            index,
            contents,
            parameters=self._parameters,
            mergebool=True,
        )

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex()

            next = self._content._carry(nextcarry, False)
            out = next._local_index(axis, depth)
            out2 = ak.contents.IndexedOptionArray(
                outindex, out, parameters=self._parameters
            )
            return out2

    def _is_subrange_equal(self, starts, stops, length, sorted=True):
        nextstarts = ak.index.Index64.empty(length, self._backend.index_nplike)
        nextstops = ak.index.Index64.empty(length, self._backend.index_nplike)

        subranges_length = ak.index.Index64.empty(1, self._backend.index_nplike)
        assert (
            self._index.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
            and nextstarts.nplike is self._backend.index_nplike
            and nextstops.nplike is self._backend.index_nplike
            and subranges_length.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_ranges_next_64",
                self._index.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                nextstarts.dtype.type,
                nextstops.dtype.type,
                subranges_length.dtype.type,
            ](
                self._index.data,
                starts.data,
                stops.data,
                length,
                nextstarts.data,
                nextstops.data,
                subranges_length.data,
            )
        )

        nextcarry = ak.index.Index64.empty(
            subranges_length[0], self._backend.index_nplike
        )
        assert (
            self._index.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
            and nextcarry.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_ranges_carry_next_64",
                self._index.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
                nextcarry.dtype.type,
            ](
                self._index.data,
                starts.data,
                stops.data,
                length,
                nextcarry.data,
            )
        )

        next = self._content._carry(nextcarry, False)
        if nextstarts.length is not unknown_length and nextstarts.length > 1:
            return next._is_subrange_equal(nextstarts, nextstops, nextstarts.length)
        else:
            return next._subranges_equal(
                nextstarts, nextstops, nextstarts.length, False
            )

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.IndexedOptionArray(
            self._index,
            self._content._numbers_to_type(name, including_unknown),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._index.length is not unknown_length and self._index.length == 0:
            return True

        projected = self.project()
        return projected._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        branch, depth = self.branch_depth

        inject_nones = (
            True if not branch and (negaxis is not None and negaxis != depth) else False
        )

        index_length = self._index.length

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        out = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
            nextoutindex = ak.index.Index64.empty(
                parents.length, self._backend.index_nplike
            )
            assert (
                nextoutindex.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
                and parents.nplike is self._backend.index_nplike
                and nextparents.nplike is self._backend.index_nplike
            )

            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_local_preparenext_64",
                    nextoutindex.dtype.type,
                    starts.dtype.type,
                    parents.dtype.type,
                    nextparents.dtype.type,
                ](
                    nextoutindex.data,
                    starts.data,
                    parents.data,
                    parents.length,
                    nextparents.data,
                    nextparents.length,
                )
            )

            return ak.contents.IndexedOptionArray.simplified(
                nextoutindex, out, parameters=self._parameters
            )

        if isinstance(out, ak.contents.ListOffsetArray):
            newnulls = ak.index.Index64.empty(
                self._index.length, self._backend.index_nplike
            )
            _len_newnulls = ak.index.Index64.empty(1, self._backend.index_nplike)
            assert (
                newnulls.nplike is self._backend.index_nplike
                and _len_newnulls.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_numnull_parents",
                    newnulls.dtype.type,
                    _len_newnulls.dtype.type,
                    self._index.dtype.type,
                ](
                    newnulls.data,
                    _len_newnulls.data,
                    self._index.data,
                    index_length,
                )
            )
            len_newnulls = self._backend.index_nplike.index_as_shape_item(
                _len_newnulls[0]
            )

            newindex = ak.index.Index64.empty(
                self._backend.index_nplike.index_as_shape_item(out._offsets[-1])
                + len_newnulls,
                self._backend.index_nplike,
            )
            newoffsets = ak.index.Index64.empty(
                out._offsets.length, self._backend.index_nplike
            )
            assert (
                newindex.nplike is self._backend.index_nplike
                and newoffsets.nplike is self._backend.index_nplike
                and out._offsets.nplike is self._backend.index_nplike
                and newnulls.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_unique_next_index_and_offsets_64",
                    newindex.dtype.type,
                    newoffsets.dtype.type,
                    out._offsets.dtype.type,
                    newnulls.dtype.type,
                ](
                    newindex.data,
                    newoffsets.data,
                    out._offsets.data,
                    newnulls.data,
                    starts.length,
                )
            )

            out = ak.contents.IndexedOptionArray.simplified(
                newindex[: newoffsets[-1]], out._content, parameters=self._parameters
            )
            return ak.contents.ListOffsetArray(
                newoffsets, out, parameters=self._parameters
            )

        if isinstance(out, ak.contents.NumpyArray):
            # We do not pass None values to the content, so it will have computed
            # the unique _non null_ values. We therefore need to account for None
            # values in the result. We do this by creating an IndexedOptionArray
            # and tacking the index -1 onto the end
            nextoutindex = ak.index.Index64.empty(
                out.length + 1,
                self._backend.index_nplike,
            )
            assert nextoutindex.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_numnull_unique_64",
                    nextoutindex.dtype.type,
                ](
                    nextoutindex.data,
                    out.length,
                )
            )

            return ak.contents.IndexedOptionArray.simplified(
                nextoutindex, out, parameters=self._parameters
            )

        if inject_nones:
            out = ak.contents.RegularArray(
                out, out.length, 0, parameters=self._parameters
            )

        return out

    def _rearrange_nextshifts(self, nextparents, shifts):
        nextshifts = ak.index.Index64.empty(
            nextparents.length, self._backend.index_nplike
        )
        assert nextshifts.nplike is self._backend.index_nplike

        if shifts is None:
            assert (
                nextshifts.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
                    nextshifts.dtype.type,
                    self._index.dtype.type,
                ](
                    nextshifts.data,
                    self._index.data,
                    self._index.length,
                )
            )
        else:
            assert (
                nextshifts.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
                and shifts.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                    nextshifts.dtype.type,
                    self._index.dtype.type,
                    shifts.dtype.type,
                ](
                    nextshifts.data,
                    self._index.data,
                    self._index.length,
                    shifts.data,
                )
            )
        return nextshifts

    def _rearrange_prepare_next(self, parents):
        assert (
            self._index.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        index_length = self._index.length
        _numnull = ak.index.Index64.empty(1, self._backend.index_nplike)
        assert _numnull.nplike is self._backend.index_nplike

        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_numnull",
                _numnull.dtype.type,
                self._index.dtype.type,
            ](
                _numnull.data,
                self._index.data,
                index_length,
            )
        )
        numnull = self._backend.index_nplike.index_as_shape_item(_numnull[0])

        next_length = index_length - numnull
        nextparents = ak.index.Index64.empty(next_length, self._backend.index_nplike)
        nextcarry = ak.index.Index64.empty(next_length, self._backend.index_nplike)
        outindex = ak.index.Index64.empty(index_length, self._backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                outindex.data,
                self._index.data,
                parents.data,
                index_length,
            )
        )
        next = self._content._carry(nextcarry, False)
        return next, nextparents, numnull, outindex

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        assert (
            starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )

        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if (not branch) and negaxis == depth:
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

        out = next._argsort_next(
            negaxis, starts, nextshifts, nextparents, outlength, ascending, stable
        )

        # `next._argsort_next` is given the non-None values. We choose to
        # sort None values to the end of the list, meaning we need to grow `out`
        # to account for these None values. First, we locate these nones within
        # their sublists
        nulls_merged = False
        nulls_index = ak.index.Index64.empty(numnull, self._backend.index_nplike)
        assert nulls_index.nplike is self._backend.index_nplike
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_index_of_nulls",
                nulls_index.dtype.type,
                self._index.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
            ](
                nulls_index.data,
                self._index.data,
                self._index.length,
                parents.data,
                starts.data,
            )
        )
        # If we wrap a NumpyArray (i.e., axis=-1), then we want `argmax` to return
        # the indices of each `None` value, rather than `None` itself.
        # We can test for this condition by seeing whether the NumpyArray of indices
        # is mergeable with our content (`out = next._argsort_next result`).
        # If so, try to concatenate them at the end of `out`.`
        nulls_index_content = ak.contents.NumpyArray(
            nulls_index.data, parameters=None, backend=self._backend
        )
        if out._mergeable_next(nulls_index_content, True):
            out = out._mergemany([nulls_index_content])
            nulls_merged = True

        nextoutindex = ak.index.Index64.empty(
            parents.length, self._backend.index_nplike
        )
        assert (
            nextoutindex.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
        )

        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_local_preparenext_64",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.data,
                starts.data,
                parents.data,
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )

        if nulls_merged:
            # awkward_IndexedArray_local_preparenext uses -1 to
            # indicate `None` values. Given that this code-path runs
            # only when the `None` value indices are explicitly stored in out,
            # we need to mapping the -1 values to their corresponding indices
            # in `out`
            assert nextoutindex.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend["awkward_Index_nones_as_index", nextoutindex.dtype.type](
                    nextoutindex.data,
                    nextoutindex.length,
                )
            )
            # Drop the option type: we have ensured that we don't have any
            # -1 values in `nextoutindex` now!
            out = out._carry(nextoutindex, False).copy(
                parameters=parameters_union(out._parameters, self._parameters)
            )
        else:
            out = ak.contents.IndexedOptionArray.simplified(
                nextoutindex, out, parameters=self._parameters
            )

        inject_nones = (
            True
            if (
                (numnull is not unknown_length and numnull > 0)
                and not branch
                and negaxis != depth
            )
            else False
        )

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout,
        # OR we are branching
        else:
            return out

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        assert (
            starts.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        out = next._sort_next(
            negaxis, starts, nextparents, outlength, ascending, stable
        )

        nextoutindex = ak.index.Index64.empty(
            parents.length, self._backend.index_nplike
        )
        assert nextoutindex.nplike is self._backend.index_nplike

        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_IndexedArray_local_preparenext_64",
                nextoutindex.dtype.type,
                starts.dtype.type,
                parents.dtype.type,
                nextparents.dtype.type,
            ](
                nextoutindex.data,
                starts.data,
                parents.data,
                parents.length,
                nextparents.data,
                nextparents.length,
            )
        )
        out = ak.contents.IndexedOptionArray.simplified(
            nextoutindex, out, parameters=self._parameters
        )

        inject_nones = True if not branch and negaxis != depth else False

        # If we want the None's at this depth to be injected
        # into the dense ([x y z None None]) rearranger result.
        # Here, we index the dense content with an index
        # that maps the values to the correct locations
        if inject_nones:
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )
        # Otherwise, if we are rearranging (e.g sorting) the contents of this layout,
        # then we do NOT want to return an optional layout
        # OR we are branching
        else:
            return out

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
        branch, depth = self.branch_depth

        next, nextparents, numnull, outindex = self._rearrange_prepare_next(parents)

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = self._rearrange_nextshifts(nextparents, shifts)
        else:
            nextshifts = None

        out = next._reduce_next(
            reducer,
            negaxis,
            starts,
            nextshifts,
            nextparents,
            outlength,
            mask,
            keepdims,
            behavior,
        )

        # If we are reducing the contents of this layout,
        # then we do NOT want to return an optional layout
        if not branch and negaxis == depth:
            return out
        else:
            if isinstance(out, ak.contents.RegularArray):
                out_content = out.content
            elif isinstance(out, ak.contents.ListOffsetArray):
                # The `outindex` that will index into `out_content` is 0-based, so we should ensure that we normalise
                # the list content to start at the first offset.
                out_content = out.content[out.offsets[0] :]
            else:
                raise AssertionError(
                    "reduce_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray or "
                    "IndexedOptionArray; "
                    f"instead, it returned {type(out).__name__}"
                )

            if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                raise AssertionError(
                    "reduce_next with unbranching depth > negaxis expects a "
                    f"ListOffsetArray whose offsets start at zero ({starts[0]})"
                )
            # In this branch, we're above the axis at which the reduction takes place.
            # `next._reduce_next` is therefore expected to return a list/regular layout
            # node. As detailed in `RegularArray._reduce_next`, `_reduce_next` wraps the
            # reduction in a list-type of length `outlength` before returning to the caller,
            # which effectively means that the reduction of *this* layout corresponds to the
            # child of the returned `next._reduce_next(...)`, i.e. `out.content`. So, we unpack
            # the returned list type and wrap its child by a new `IndexedOptionArray`, before
            # re-wrapping the result to have the length and starts requested by the caller.
            if starts.length is unknown_length:
                outoffsets = ak.index.Index64.empty(
                    unknown_length, self._backend.index_nplike
                )
            else:
                outoffsets = ak.index.Index64.empty(
                    starts.length + 1,
                    self._backend.index_nplike,
                )
            assert (
                outoffsets.nplike is self._backend.index_nplike
                and starts.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_reduce_next_fix_offsets_64",
                    outoffsets.dtype.type,
                    starts.dtype.type,
                ](
                    outoffsets.data,
                    starts.data,
                    starts.length,
                    outindex.length,
                )
            )

            # Apply `outindex` to appropriate content
            inner = ak.contents.IndexedOptionArray.simplified(
                outindex, out_content, parameters=self._parameters
            )

            # Re-wrap content
            return ak.contents.ListOffsetArray(
                outoffsets, inner, parameters=self._parameters
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex()
            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, axis, depth
            )
            return IndexedOptionArray.simplified(outindex, out, parameters=parameters)

    def _validity_error(self, path):
        if self.parameter("__array__") == "categorical":
            if not ak._do.is_unique(self._content):
                return f'at {path} ("{type(self)}"): __array__ = "categorical" requires contents to be unique'

        assert self.index.nplike is self._backend.index_nplike
        error = self._backend["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.data, self.index.length, self._content.length, True
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
        return self.index._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            mask = ak.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak.index.Index64.empty(mask.length, self._backend.index_nplike)
            assert (
                index.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    mask.dtype.type,
                ](index.data, mask.data, mask.length)
            )
            next = self.project()._pad_none(target, axis, depth, clip)
            return ak.contents.IndexedOptionArray.simplified(
                index, next, parameters=self._parameters
            )
        else:
            return ak.contents.IndexedOptionArray(
                self._index,
                self._content._pad_none(target, axis, depth, clip),
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
        index = numpy.asarray(self._index.data, copy=True)
        this_validbytes = self.mask_as_bool(valid_when=True)
        index[~this_validbytes] = 0

        if self.parameter("__array__") == "categorical":
            # The new IndexedArray will have this parameter, but the rest
            # will be in the AwkwardArrowType.mask_parameters.
            next_parameters = {"__array__": "categorical"}
        else:
            next_parameters = None

        next = ak.contents.IndexedArray(
            ak.index.Index(index), self._content, parameters=next_parameters
        )
        return next._to_arrow(
            pyarrow,
            self,
            ak._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        return self.to_ByteMaskedArray(True)._to_cudf(cudf, mask, length)

    def _to_backend_array(self, allow_missing, backend):
        nplike = backend.nplike
        index_nplike = backend.index_nplike

        content = self.project()._to_backend_array(allow_missing, backend)
        shape = (self.length, *content.shape[1:])
        data = nplike.empty(shape, dtype=content.dtype)
        mask0 = index_nplike.asarray(self.mask_as_bool(valid_when=False)).view(np.bool_)
        if index_nplike.any(mask0):
            if allow_missing:
                mask = index_nplike.broadcast_to(
                    index_nplike.reshape(mask0, (shape[0],) + (1,) * (len(shape) - 1)),
                    shape,
                )
                if isinstance(content, nplike.ma.MaskedArray):
                    mask1 = index_nplike.ma.getmaskarray(content)
                    mask = index_nplike.asarray(mask, copy=True)
                    mask[index_nplike.logical_not(mask0)] |= mask1

                data[index_nplike.logical_not(mask0)] = content

                if content.dtype.names is not None:
                    missing_data = tuple(
                        create_missing_data(each_dtype, backend)
                        for each_dtype, _ in content.dtype.fields.values()
                    )
                else:
                    missing_data = create_missing_data(content.dtype, backend)

                data[mask0] = missing_data

                return nplike.ma.MaskedArray(data, mask)
            else:
                raise ValueError(
                    "Content.to_nplike cannot convert 'None' values to "
                    "np.ma.MaskedArray unless the "
                    "'allow_missing' parameter is set to True"
                )
        else:
            if allow_missing:
                return nplike.ma.MaskedArray(content)
            else:
                return content

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        branch, depth = self.branch_depth
        if branch or options["drop_nones"] or depth > 1:
            return self.project()._remove_structure(backend, options)
        else:
            return [self]

    def _drop_none(self) -> Content:
        return self.project()

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
            and self._index.length != 0
        ):
            npindex = self._index.data
            npselect = npindex >= 0
            if self._backend.index_nplike.any(npselect):
                indexmin = self._backend.index_nplike.min(npindex[npselect])
                index = ak.index.Index(
                    npindex - indexmin, nplike=self._backend.index_nplike
                )
                content = self._content[indexmin : npindex.max() + 1]
            else:
                index, content = self._index, self._content
        else:
            if not self._backend.nplike.known_data:
                self._touch_data(recursive=False)
            index, content = self._index, self._content

        if options["return_array"]:
            if options["return_simplified"]:
                make = IndexedOptionArray.simplified
            else:
                make = IndexedOptionArray

            def continuation():
                return make(
                    index,
                    content._recursively_apply(
                        action,
                        depth,
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
                    depth,
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
        if self.parameter("__array__") == "categorical":
            content = self._content.to_packed(True) if recursive else self._content
            return ak.contents.IndexedOptionArray(
                self._index, content, parameters=self._parameters
            )

        else:
            index_nplike = self._backend.index_nplike
            original_index = self._index.data
            is_none = original_index < 0
            num_none = index_nplike.count_nonzero(is_none)
            new_index = index_nplike.empty(self._index.length, dtype=self._index.dtype)
            new_index[is_none] = -1
            new_index[~is_none] = index_nplike.arange(
                index_nplike.shape_item_as_index(new_index.size - num_none),
                dtype=self._index.dtype,
            )
            projected = self.project()
            return ak.contents.IndexedOptionArray(
                ak.index.Index(new_index, nplike=self._backend.index_nplike),
                projected.to_packed(True) if recursive else projected,
                parameters=self._parameters,
            )

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        not_missing = index >= 0
        content = ak.to_backend(self._content, "cpu", highlevel=False)
        nextcontent = content._carry(ak.index.Index(index[not_missing]), False)
        out = nextcontent._to_list(behavior, json_conversions)

        for i, isvalid in enumerate(not_missing):
            if not isvalid:
                out.insert(i, None)

        return out

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        index = self._index.to_nplike(backend.index_nplike)
        return IndexedOptionArray(index, content, parameters=self._parameters)

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and self._index.is_equal_to(other.index, index_dtype, numpyarray)
            and self._content._is_equal_to(
                other.content, index_dtype, numpyarray, all_parameters
            )
        )

    def _trim(self) -> Self:
        nplike = self._backend.index_nplike

        if not nplike.known_data or self._index.length == 0:
            return self

        idx_buf = nplike.asarray(self._index.data, copy=True)
        only_positive = idx_buf >= 0

        # no positive index at all
        if not nplike.any(only_positive):
            return self

        min_idx = nplike.min(idx_buf[only_positive])
        max_idx = nplike.max(idx_buf[only_positive])
        idx_buf[only_positive] -= min_idx
        index = Index(idx_buf)

        # left and right trim
        content = self._content._getitem_range(min_idx, max_idx + 1)
        return IndexedOptionArray(index, content, parameters=self._parameters)


def create_missing_data(dtype, backend):
    """Create missing data based on the input dtype

    Missing data are represented differently based on the Numpy array
    dtype, this function returns the proper missing data representation
    given the input dtype
    """
    if np.issubdtype(dtype, np.bool_):
        return False
    elif np.issubdtype(dtype, np.floating):
        return np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        return np.nan + np.nan * 1j
    elif np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype.type, np.datetime64) or np.issubdtype(
        dtype.type, np.timedelta64
    ):
        return backend.nplike.asarray([np.iinfo(np.int64).max], dtype=dtype)[0]
    elif np.issubdtype(dtype, np.str_):
        return ""
    elif np.issubdtype(dtype, np.bytes_):
        return b""
    else:
        raise AssertionError(f"unrecognized dtype: {dtype}")
