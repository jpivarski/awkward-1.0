# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.indexedmeta import IndexedMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import TypeTracer
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
from awkward.forms.indexedform import IndexedForm
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem
    from awkward.contents.indexedoptionarray import IndexedOptionArray

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class IndexedArray(IndexedMeta[Content], Content):
    """
    IndexedArray is a general-purpose tool for *lazily* changing the order of
    and/or duplicating some `content` with a
    [np.take](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)
    over the integer buffer `index.

    It has many uses:

    * representing a lazily applied slice.
    * simulating pointers into another collection.
    * emulating the dictionary encoding of Apache Arrow and Parquet.

    If the `__array__` parameter is `"categorical"`, the contents must be unique.
    Some operations are optimized (for instance, `==` only compares `index` integers)
    and the array can be converted to and from Arrow/Parquet's dictionary encoding.

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class IndexedArray(Content):
            def __init__(self, index, content):
                assert isinstance(index, (Index32, IndexU32, Index64))
                assert isinstance(content, Content)
                for x in index:
                    assert 0 <= x < len(content)  # index[i] must not be negative
                self.index = index
                self.content = content

            def __len__(self):
                return len(self.index)

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    return self.content[self.index[where]]

                elif isinstance(where, slice) and where.step is None:
                    return IndexedArray(
                        self.index[where.start : where.stop], self.content
                    )

                elif isinstance(where, str):
                    return IndexedArray(self.index, self.content[where])

                else:
                    raise AssertionError(where)
    """

    def __init__(self, index, content, *, parameters=None):
        if not (
            isinstance(index, Index)
            and index.dtype
            in (
                np.dtype(np.int32),
                np.dtype(np.uint32),
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

    form_cls: Final = IndexedForm

    def copy(self, index=UNSET, content=UNSET, *, parameters=UNSET):
        return IndexedArray(
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
            return content._carry(index, allow_lazy=False).copy(
                parameters=parameters_union(content._parameters, parameters)
            )

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
            if isinstance(content, ak.contents.IndexedArray):
                return ak.contents.IndexedArray(
                    result,
                    content.content,
                    parameters=parameters_union(content._parameters, parameters),
                )
            else:
                return ak.contents.IndexedOptionArray(
                    result,
                    content.content,
                    parameters=parameters_union(content._parameters, parameters),
                )

        else:
            return cls(index, content, parameters=parameters)

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> IndexedForm:
        form_key = getkey(self)
        return self.form_cls(
            self._index.form,
            self._content._form_with_key(getkey),
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> IndexedForm:
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
        return IndexedArray(
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
        out = [indent, pre, "<IndexedArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</IndexedArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self) -> IndexedOptionArray:
        return ak.contents.IndexedOptionArray(
            self._index, self._content, parameters=self._parameters
        )

    def mask_as_bool(self, valid_when: bool = True) -> ArrayLike:
        if valid_when:
            return self._index.data >= 0
        else:
            return self._index.data < 0

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
            return self._content._getitem_at(where)

        if where < 0:
            where += self.length
        if self._backend.nplike.known_data and not 0 <= where < self.length:
            raise ak._errors.index_error(self, where)
        return self._content._getitem_at(self._index[where])

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        return IndexedArray(
            self._index[start:stop], self._content, parameters=self._parameters
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return IndexedArray.simplified(
            self._index,
            self._content._getitem_field(where, only_fields),
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return IndexedArray.simplified(
            self._index,
            self._content._getitem_fields(where, only_fields),
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> IndexedArray:
        assert isinstance(carry, ak.index.Index)

        try:
            nextindex = self._index[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return IndexedArray(nextindex, self._content, parameters=self._parameters)

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if self._backend.nplike.known_data and slicestarts.length != self.length:
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                f"cannot fit jagged slice with length {slicestarts.length} into {type(self).__name__} of size {self.length}",
            )

        nextcarry = ak.index.Index64.empty(self.length, self._backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._maybe_index_error(
            self._backend[
                "awkward_IndexedArray_getitem_nextcarry",
                nextcarry.dtype.type,
                self._index.dtype.type,
            ](
                nextcarry.data,
                self._index.data,
                self._index.length,
                self._content.length,
            ),
            slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
        )
        # an eager carry (allow_lazy = false) to avoid infinite loop (unproven)
        next = self._content._carry(nextcarry, False)
        return next._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

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

            nextcarry = ak.index.Index64.empty(
                self._index.length, self._backend.index_nplike
            )
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._maybe_index_error(
                self._backend[
                    "awkward_IndexedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.data,
                    self._index.data,
                    self._index.length,
                    self._content.length,
                ),
                slicer=head,
            )

            next = self._content._carry(nextcarry, False)
            return next._getitem_next(head, tail, advanced)

        elif isinstance(head, str):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
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
            nextcarry = ak.index.Index64.empty(self.length, self._backend.index_nplike)
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._index.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._index.dtype.type,
                ](
                    nextcarry.data,
                    self._index.data,
                    self._index.length,
                    self._content.length,
                )
            )
            next = self._content._carry(nextcarry, False)
            return next.copy(
                parameters=parameters_union(
                    next._parameters,
                    self._parameters,
                    exclude=(("__array__", "categorical"),),
                )
            )

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError("axis=0 not allowed for flatten")

        else:
            return self.project()._offsets_and_flattened(axis, depth)

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
            (theirlength + mylength), self._backend.index_nplike
        )

        content = other._mergemany([self._content])

        # Fill `index` with a range starting at zero, up to `theirlength`
        assert index.nplike is self._backend.index_nplike
        self._backend.maybe_kernel_error(
            self._backend["awkward_IndexedArray_fill_count", index.dtype.type](
                index.data,
                0,
                theirlength,
                0,
            )
        )

        # Fill remaining indices
        assert index.nplike is self._backend.index_nplike
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

        return ak.contents.IndexedArray.simplified(
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
        nextindex = ak.index.Index64.empty(
            total_length, nplike=self._backend.index_nplike
        )

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

        # FIXME: support categorical merging?
        if parameters is not None and parameters.get("__array__") == "categorical":
            raise NotImplementedError(
                "merging categorical arrays is currently not implemented. "
                "Use `ak.enforce_type` to drop the categorical type and use general merging."
            )

        # Options win out!
        if any(x.is_option for x in head):
            next = ak.contents.IndexedOptionArray(
                nextindex, nextcontent, parameters=parameters
            )
        else:
            next = ak.contents.IndexedArray(
                nextindex, nextcontent, parameters=parameters
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
        return IndexedArray(
            self._index, self._content._fill_none(value), parameters=self._parameters
        )

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            return self.project()._local_index(axis, depth)

    def _unique_index(self, index, sorted=True):
        next = ak.index.Index64.zeros(self.length, nplike=self._backend.index_nplike)
        length = ak.index.Index64.zeros(1, nplike=self._backend.index_nplike)

        if not sorted:
            next = self._index
            offsets = ak.index.Index64.empty(2, self._backend.index_nplike)
            offsets[0] = 0
            offsets[1] = next.length
            assert (
                next.nplike is self._backend.index_nplike
                and offsets.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_sort",
                    next.dtype.type,
                    next.dtype.type,
                    offsets.dtype.type,
                ](
                    next.data,
                    next.data,
                    offsets[1],
                    offsets.data,
                    2,
                    offsets[1],
                    True,
                    False,
                )
            )

        assert (
            self._index.nplike is self._backend.index_nplike
            and next.nplike is self._backend.index_nplike
            and length.nplike is self._backend.index_nplike
        )

        next = ak.index.Index64(
            self._backend.index_nplike.unique_values(self._index),
            nplike=self._backend.index_nplike,
        )
        length[0] = next.data.size

        return next[0 : length[0]]

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.IndexedArray(
            self._index,
            self._content._numbers_to_type(name, including_unknown),
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._index.length is not unknown_length and self._index.length == 0:
            return True

        nextindex = self._unique_index(self._index)

        if len(nextindex) != len(self._index):
            return False

        next = self._content._carry(nextindex, False)
        return next._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        if self._index.length is not unknown_length and self._index.length == 0:
            return self

        branch, depth = self.branch_depth

        index_length = self._index.length
        parents_length = parents.length
        next_length = index_length

        nextcarry = ak.index.Index64.empty(index_length, self._backend.index_nplike)
        nextparents = ak.index.Index64.empty(index_length, self._backend.index_nplike)
        outindex = ak.index.Index64.empty(index_length, self._backend.index_nplike)
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
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
        unique = next._unique(
            negaxis,
            starts,
            nextparents,
            outlength,
        )

        if branch or (negaxis is not None and negaxis != depth):
            nextoutindex = ak.index.Index64.empty(
                parents_length, self._backend.index_nplike
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
                    parents_length,
                    nextparents.data,
                    next_length,
                )
            )

            return ak.contents.IndexedOptionArray.simplified(
                nextoutindex, unique, parameters=self._parameters
            )

        if not branch and negaxis == depth:
            return unique
        else:
            if isinstance(unique, ak.contents.RegularArray):
                unique = unique.to_ListOffsetArray64(True)

            if isinstance(unique, ak.contents.ListOffsetArray):
                if starts.nplike.known_data and starts.length > 0 and starts[0] != 0:
                    raise AssertionError(
                        "reduce_next with unbranching depth > negaxis expects a "
                        f"ListOffsetArray64 whose offsets start at zero ({starts[0]})"
                    )

                outoffsets = ak.index.Index64.empty(
                    starts.length + 1, self._backend.index_nplike
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
                        self._index.length,
                    )
                )

                tmp = ak.contents.IndexedArray(
                    outindex, unique._content, parameters=None
                )

                return ak.contents.ListOffsetArray(outoffsets, tmp, parameters=None)

            elif isinstance(unique, ak.contents.NumpyArray):
                nextoutindex = ak.index.Index64(
                    self._backend.index_nplike.arange(unique.length, dtype=np.int64),
                    nplike=self._backend.index_nplike,
                )
                return ak.contents.IndexedOptionArray.simplified(
                    nextoutindex, unique, parameters=self._parameters
                )

        raise NotImplementedError

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        next = self._content._carry(self._index, False)
        return next._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        next = self._content._carry(self._index, False)
        return next._sort_next(negaxis, starts, parents, outlength, ascending, stable)

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return self.project()._combinations(
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
        next = self._content._carry(self._index, False)
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

    def _validity_error(self, path):
        if self.parameter("__array__") == "categorical":
            if not ak._do.is_unique(self._content):
                return f'at {path} ("{type(self)}"): __array__ = "categorical" requires contents to be unique'
        error = self._backend["awkward_IndexedArray_validity", self.index.dtype.type](
            self.index.data, self.index.length, self._content.length, False
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
            return self.project()._pad_none(target, axis, depth, clip)
        else:
            return ak.contents.IndexedArray(
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
        if (
            not options["categorical_as_dictionary"]
            and self.parameter("__array__") == "categorical"
        ):
            next_parameters = dict(self._parameters)
            del next_parameters["__array__"]
            next = IndexedArray(self._index, self._content, parameters=next_parameters)
            return next._to_arrow(pyarrow, mask_node, validbytes, length, options)

        index = self._index.raw(numpy)

        if self.parameter("__array__") == "categorical":
            dictionary = self._content._to_arrow(
                pyarrow, None, None, self._content.length, options
            )
            out = pyarrow.DictionaryArray.from_arrays(
                index,
                dictionary,
                None if validbytes is None else ~validbytes,
            )
            if options["extensionarray"]:
                return ak._connect.pyarrow.AwkwardArrowArray.from_storage(
                    ak._connect.pyarrow.to_awkwardarrow_type(
                        out.type,
                        options["extensionarray"],
                        options["record_is_scalar"],
                        mask_node,
                        self,
                    ),
                    out,
                )
            else:
                return out

        else:
            if self._content.length is not unknown_length and self._content.length == 0:
                # IndexedOptionArray._to_arrow replaces -1 in the index with 0. So behind
                # every masked value is self._content[0], unless self._content.length == 0.
                # In that case, don't call self._content[index]; it's empty anyway.
                next = self._content
            else:
                next = self._content._carry(ak.index.Index(index), False)

            next2 = next.copy(
                parameters=parameters_union(next._parameters, self._parameters)
            )
            return next2._to_arrow(pyarrow, mask_node, validbytes, length, options)

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        if self._content.length is not unknown_length and self._content.length == 0:
            # IndexedOptionArray._to_arrow replaces -1 in the index with 0. So behind
            # every masked value is self._content[0], unless self._content.length == 0.
            # In that case, don't call self._content[index]; it's empty anyway.
            next = self._content
        else:
            next = self._content._carry(self._index, False)
        return next._to_cudf(cudf, None, len(next))

    def _to_backend_array(self, allow_missing, backend):
        return self.project()._to_backend_array(allow_missing, backend)

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        return self.project()._remove_structure(backend, options)

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
            indexmin = self._backend.index_nplike.min(npindex)
            index = ak.index.Index(
                npindex - indexmin, nplike=self._backend.index_nplike
            )
            content = self._content[indexmin : npindex.max() + 1]
        else:
            if not self._backend.nplike.known_data:
                self._touch_data(recursive=False)
            index, content = self._index, self._content

        if options["return_array"]:
            if options["return_simplified"]:
                make = IndexedArray.simplified
            else:
                make = IndexedArray

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
            return IndexedArray(self._index, content, parameters=self._parameters)
        else:
            projected = self.project()
            return projected.to_packed(True) if recursive else projected

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        index = self._index.raw(numpy)
        nextcontent = self._content._carry(ak.index.Index(index), False)
        return nextcontent._to_list(behavior, json_conversions)

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        index = self._index.to_nplike(backend.index_nplike)
        return IndexedArray(index, content, parameters=self._parameters)

    def _push_inside_record_or_project(self) -> Self | ak.contents.RecordArray:
        if self.content.is_record:
            return ak.contents.RecordArray(
                contents=[
                    ak.contents.IndexedArray.simplified(self._index, c)
                    for c in self.content.contents
                ],
                fields=self.content._fields,
                length=self.length,
                backend=self._backend,
                parameters=parameters_union(self.content._parameters, self._parameters),
            )
        else:
            return self.project()

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
        min_idx = nplike.min(idx_buf)
        max_idx = nplike.max(idx_buf)
        idx_buf -= min_idx
        index = Index(idx_buf)

        # left and right trim
        content = self._content._getitem_range(min_idx, max_idx + 1)
        return IndexedArray(index, content, parameters=self._parameters)
