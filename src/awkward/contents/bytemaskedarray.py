# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.bytemaskedmeta import ByteMaskedMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import MaybeNone, TypeTracer
from awkward._nplikes.virtual import VirtualArray
from awkward._parameters import (
    parameters_intersect,
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
from awkward.forms.bytemaskedform import ByteMaskedForm
from awkward.forms.form import Form, FormKeyPathT
from awkward.index import Index

if TYPE_CHECKING:
    from awkward._slicing import SliceItem
    from awkward.contents import IndexedOptionArray

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@final
class ByteMaskedArray(ByteMaskedMeta[Content], Content):
    """
    The ByteMaskedArray implements an #ak.types.OptionType with two aligned
    buffers, a boolean `mask` and `content`. At any element `i` where
    `mask[i] == valid_when`, the value can be found at `content[i]`. If
    `mask[i] != valid_when`, the value is missing (None).

    This is equivalent to NumPy's
    [masked arrays](https://docs.scipy.org/doc/numpy/reference/maskedarray.html)
    if `valid_when=False`.

    There is no Apache Arrow equivalent because Arrow
    [uses bitmaps](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps)
    to mask all node types.

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class ByteMaskedArray(Content):
            def __init__(self, mask, content, valid_when):
                assert isinstance(mask, Index8)
                assert isinstance(content, Content)
                assert isinstance(valid_when, bool)
                assert len(mask) <= len(content)
                self.mask = mask
                self.content = content
                self.valid_when = valid_when

            def __len__(self):
                return len(self.mask)

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    if self.mask[where] == self.valid_when:
                        return self.content[where]
                    else:
                        return None

                elif isinstance(where, slice) and where.step is None:
                    return ByteMaskedArray(
                        self.mask[where.start : where.stop],
                        self.content[where.start : where.stop],
                        valid_when=self.valid_when,
                    )

                elif isinstance(where, str):
                    return ByteMaskedArray(
                        self.mask, self.content[where], valid_when=self.valid_when
                    )

                else:
                    raise AssertionError(where)
    """

    def __init__(self, mask, content, valid_when, *, parameters=None):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise TypeError(
                f"{type(self).__name__} 'mask' must be an Index with dtype=int8, not {mask!r}"
            )
        if not isinstance(content, Content):
            raise TypeError(
                f"{type(self).__name__} 'content' must be a Content subtype, not {content!r}"
            )
        if content.is_union or content.is_indexed or content.is_option:
            raise TypeError(
                "{0} cannot contain a union-type, option-type, or indexed 'content' ({1}); try {0}.simplified instead".format(
                    type(self).__name__, type(content).__name__
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                f"{type(self).__name__} 'valid_when' must be boolean, not {valid_when!r}"
            )
        if (
            content.backend.index_nplike.known_data
            and mask.length is not unknown_length
            and content.length is not unknown_length
            and mask.length > content.length
        ):
            raise ValueError(
                f"{type(self).__name__} len(mask) ({mask.length}) must be <= len(content) ({content.length})"
            )

        assert mask.nplike is content.backend.index_nplike

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(parameters, content.backend)

    @property
    def mask(self):
        return self._mask

    @property
    def valid_when(self):
        return self._valid_when

    form_cls: Final = ByteMaskedForm

    def copy(self, mask=UNSET, content=UNSET, valid_when=UNSET, *, parameters=UNSET):
        return ByteMaskedArray(
            self._mask if mask is UNSET else mask,
            self._content if content is UNSET else content,
            self._valid_when if valid_when is UNSET else valid_when,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            mask=copy.deepcopy(self._mask, memo),
            content=copy.deepcopy(self._content, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(
        cls,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
    ):
        if content.is_union or content.is_indexed or content.is_option:
            backend = content.backend
            index = ak.index.Index64.empty(mask.length, nplike=backend.index_nplike)
            backend.maybe_kernel_error(
                backend[
                    "awkward_ByteMaskedArray_toIndexedOptionArray",
                    index.dtype.type,
                    mask.dtype.type,
                ](
                    index.data,
                    mask.data,
                    mask.length,
                    valid_when,
                ),
            )
            if content.is_union:
                return content._union_of_optionarrays(index, parameters)
            else:
                return ak.contents.IndexedOptionArray.simplified(
                    index, content, parameters=parameters
                )
        else:
            return cls(mask, content, valid_when, parameters=parameters)

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> ByteMaskedForm:
        form_key = getkey(self)
        return self.form_cls(
            self._mask.form,
            self._content._form_with_key(getkey),
            self._valid_when,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> ByteMaskedForm:
        return self.form_cls(
            self._mask.form,
            self._content._form_with_key_path((*path, None)),
            self._valid_when,
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
        key = getkey(self, form, "mask")
        container[key] = ak._util.native_to_byteorder(
            self._mask.raw(backend.index_nplike), byteorder
        )
        self._content._to_buffers(form.content, getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        tt = TypeTracer.instance()
        mask = self._mask.to_nplike(tt)
        return ByteMaskedArray(
            mask.forget_length() if forget_length else mask,
            self._content._to_typetracer(forget_length),
            self._valid_when,
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        self._mask._touch_data()
        if recursive:
            self._content._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        self._mask._touch_shape()
        if recursive:
            self._content._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._mask.length

    def _forget_length(self):
        return ByteMaskedArray(
            self._mask.forget_length(),
            self._content,
            self._valid_when,
            parameters=self._parameters,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray valid_when="]
        out.append(repr(json.dumps(self._valid_when)))
        out.append(" len=")
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</ByteMaskedArray>")
        out.append(post)
        return "".join(out)

    def to_IndexedOptionArray64(self) -> IndexedOptionArray:
        index = ak.index.Index64.empty(
            self._mask.length, nplike=self._backend.index_nplike
        )
        assert (
            index.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ByteMaskedArray_toIndexedOptionArray",
                index.dtype.type,
                self._mask.dtype.type,
            ](
                index.data,
                self._mask.data,
                self._mask.length,
                self._valid_when,
            ),
        )
        return ak.contents.IndexedOptionArray(
            index, self._content, parameters=self._parameters
        )

    def to_ByteMaskedArray(self, valid_when):
        if valid_when == self._valid_when:
            return self
        else:
            return ByteMaskedArray(
                ak.index.Index8(
                    self._backend.index_nplike.astype(
                        self._backend.index_nplike.logical_not(
                            self._backend.index_nplike.astype(
                                self._mask.data, dtype=np.bool_
                            )
                        ),
                        dtype=np.int8,
                    )
                ),
                self._content,
                valid_when,
                parameters=self._parameters,
            )

    def to_BitMaskedArray(self, valid_when, lsb_order):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            if self._backend.nplike.known_data:
                excess_length = int(math.ceil(self.length / 8.0))
            else:
                excess_length = unknown_length
            return ak.contents.BitMaskedArray(
                ak.index.IndexU8(
                    self._backend.nplike.empty(excess_length, dtype=np.uint8)
                ),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                parameters=self._parameters,
            )

        else:
            bit_order = "little" if lsb_order else "big"
            bytemask = self.mask_as_bool(valid_when).view(np.uint8)
            bitmask = self.backend.index_nplike.packbits(bytemask, bitorder=bit_order)

            return ak.contents.BitMaskedArray(
                ak.index.IndexU8(bitmask),
                self._content,
                valid_when,
                self.length,
                lsb_order,
                parameters=self._parameters,
            )

    def mask_as_bool(self, valid_when=None):
        if valid_when is None:
            valid_when = self._valid_when

        if valid_when == self._valid_when:
            return self._mask.raw(self._backend.index_nplike) != 0
        else:
            return self._mask.raw(self._backend.index_nplike) != 1

    def _getitem_nothing(self):
        return self._content._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder = isinstance(self._mask, PlaceholderArray)
        if is_placeholder:
            return True
        return self._content._is_getitem_at_placeholder()

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual = (
            isinstance(self._mask, VirtualArray) and not self._mask.is_materialized
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
        if self._mask[where] == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        return ByteMaskedArray(
            self._mask[start:stop],
            self._content._getitem_range(start, stop),
            self._valid_when,
            parameters=self._parameters,
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ByteMaskedArray.simplified(
            self._mask,
            self._content._getitem_field(where, only_fields),
            self._valid_when,
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return ByteMaskedArray.simplified(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)

        try:
            nextmask = self._mask[carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return ByteMaskedArray.simplified(
            nextmask,
            self._content._carry(carry, allow_lazy),
            self._valid_when,
            parameters=self._parameters,
        )

    def _nextcarry_outindex(self) -> tuple[int, ak.index.Index64, ak.index.Index64]:
        _numnull = ak.index.Index64.empty(1, nplike=self._backend.index_nplike)

        assert (
            _numnull.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ByteMaskedArray_numnull",
                _numnull.dtype.type,
                self._mask.dtype.type,
            ](
                _numnull.data,
                self._mask.data,
                self._mask.length,
                self._valid_when,
            )
        )
        numnull = self._backend.index_nplike.index_as_shape_item(_numnull[0])
        nextcarry = ak.index.Index64.empty(
            self.length - numnull,
            nplike=self._backend.index_nplike,
        )
        outindex = ak.index.Index64.empty(
            self.length, nplike=self._backend.index_nplike
        )
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
                nextcarry.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
            ](
                nextcarry.data,
                outindex.data,
                self._mask.raw(self._backend.nplike),
                self._mask.length,
                self._valid_when,
            )
        )
        return numnull, nextcarry, outindex

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if (
            slicestarts.nplike.known_data
            and self._backend.nplike.known_data
            and slicestarts.length != self.length
        ):
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
            and reducedstarts.nplike is self._backend.nplike
            and reducedstops.nplike is self._backend.nplike
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
            _, nextcarry, outindex = self._nextcarry_outindex()
            next = self._content._carry(nextcarry, True)
            out = next._getitem_next(head, tail, advanced)
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

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
        mask_length = self._mask.length
        _numnull = ak.index.Index64.zeros(1, nplike=self._backend.index_nplike)

        if mask is not None:
            if self._backend.nplike.known_data and mask_length != mask.length:
                raise ValueError(
                    f"mask length ({mask.length}) is not equal to {type(self).__name__} length ({mask_length})"
                )

            nextmask = ak.index.Index8.empty(
                mask_length, nplike=self._backend.index_nplike
            )
            assert (
                nextmask.nplike is self._backend.index_nplike
                and mask.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ByteMaskedArray_overlay_mask",
                    nextmask.dtype.type,
                    mask.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextmask.data,
                    mask.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
                )
            )
            valid_when = False
            next = ByteMaskedArray(
                nextmask, self._content, valid_when, parameters=self._parameters
            )
            return next.project()

        else:
            assert (
                _numnull.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ByteMaskedArray_numnull",
                    _numnull.dtype.type,
                    self._mask.dtype.type,
                ](
                    _numnull.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
                )
            )
            numnull = self._backend.index_nplike.index_as_shape_item(_numnull[0])
            nextcarry = ak.index.Index64.empty(
                mask_length - numnull, nplike=self._backend.index_nplike
            )
            assert (
                nextcarry.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_ByteMaskedArray_getitem_nextcarry",
                    nextcarry.dtype.type,
                    self._mask.dtype.type,
                ](
                    nextcarry.data,
                    self._mask.data,
                    mask_length,
                    self._valid_when,
                )
            )

            return self._content._carry(nextcarry, False)

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
                    nplike=self._backend.index_nplike,
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

    def _reverse_merge(self, other):
        return self.to_IndexedOptionArray64()._reverse_merge(other)

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self

        if all(
            isinstance(x, ByteMaskedArray) and x._valid_when == self._valid_when
            for x in others
        ):
            parameters = self._parameters
            self_length_scalar = self._backend.index_nplike.shape_item_as_index(
                self.length
            )
            masks = [self._mask.data[:self_length_scalar]]
            tail_contents = []
            length = 0
            for x in others:
                length_scalar = self._backend.index_nplike.shape_item_as_index(x.length)
                parameters = parameters_intersect(parameters, x._parameters)
                masks.append(x._mask.data[:length_scalar])
                tail_contents.append(x._content[:length_scalar])
                length += x.length

            return ByteMaskedArray(
                ak.index.Index8(self._backend.nplike.concat(masks)),
                self._content[:self_length_scalar]._mergemany(tail_contents),
                self._valid_when,
                parameters=parameters,
            )

        else:
            return self.to_IndexedOptionArray64()._mergemany(others)

    def _fill_none(self, value: Content) -> Content:
        return self.to_IndexedOptionArray64()._fill_none(value)

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            _, nextcarry, outindex = self._nextcarry_outindex()

            next = self._content._carry(nextcarry, False)
            out = next._local_index(axis, depth)
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=self._parameters
            )

    def _numbers_to_type(self, name, including_unknown):
        return ak.contents.ByteMaskedArray(
            self._mask,
            self._content._numbers_to_type(name, including_unknown),
            self._valid_when,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._mask.length is not unknown_length and self._mask.length == 0:
            return True
        return self.to_IndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._mask.length is not unknown_length and self._mask.length == 0:
            return self
        return self.to_IndexedOptionArray64()._unique(
            negaxis, starts, parents, outlength
        )

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        return self.to_IndexedOptionArray64()._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        return self.to_IndexedOptionArray64()._sort_next(
            negaxis, starts, parents, outlength, ascending, stable
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        if n < 1:
            raise ValueError("in combinations, 'n' must be at least 1")
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            _, nextcarry, outindex = self._nextcarry_outindex()

            next = self._content._carry(nextcarry, True)
            out = next._combinations(
                n, replacement, recordlookup, parameters, axis, depth
            )
            return ak.contents.IndexedOptionArray.simplified(
                outindex, out, parameters=parameters
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
        mask_length = self._mask.length

        _numnull = ak.index.Index64.empty(1, nplike=self._backend.index_nplike)
        assert (
            _numnull.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ByteMaskedArray_numnull",
                _numnull.dtype.type,
                self._mask.dtype.type,
            ](
                _numnull.data,
                self._mask.data,
                mask_length,
                self._valid_when,
            )
        )
        numnull = self._backend.index_nplike.index_as_shape_item(_numnull[0])

        next_length = mask_length - numnull
        nextcarry = ak.index.Index64.empty(
            next_length, nplike=self._backend.index_nplike
        )
        nextparents = ak.index.Index64.empty(
            next_length, nplike=self._backend.index_nplike
        )
        outindex = ak.index.Index64.empty(
            mask_length, nplike=self._backend.index_nplike
        )
        assert (
            nextcarry.nplike is self._backend.index_nplike
            and nextparents.nplike is self._backend.index_nplike
            and outindex.nplike is self._backend.index_nplike
            and self._mask.nplike is self._backend.index_nplike
            and parents.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_ByteMaskedArray_reduce_next_64",
                nextcarry.dtype.type,
                nextparents.dtype.type,
                outindex.dtype.type,
                self._mask.dtype.type,
                parents.dtype.type,
            ](
                nextcarry.data,
                nextparents.data,
                outindex.data,
                self._mask.data,
                parents.data,
                mask_length,
                self._valid_when,
            )
        )

        branch, depth = self.branch_depth

        if reducer.needs_position and (not branch and negaxis == depth):
            nextshifts = ak.index.Index64.empty(
                next_length, nplike=self._backend.index_nplike
            )
            if shifts is None:
                assert (
                    nextshifts.nplike is self._backend.index_nplike
                    and self._mask.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                    ](
                        nextshifts.data,
                        self._mask.data,
                        mask_length,
                        self._valid_when,
                    )
                )
            else:
                assert (
                    nextshifts.nplike is self._backend.index_nplike
                    and self._mask.nplike is self._backend.index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
                        nextshifts.dtype.type,
                        self._mask.dtype.type,
                        shifts.dtype.type,
                    ](
                        nextshifts.data,
                        self._mask.data,
                        mask_length,
                        self._valid_when,
                        shifts.data,
                    )
                )
        else:
            nextshifts = None

        next = self._content._carry(nextcarry, False)

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
                raise ValueError(
                    "reduce_next with unbranching depth > negaxis is only "
                    "expected to return RegularArray or ListOffsetArray64; "
                    "instead, it returned " + out
                )

            outoffsets = ak.index.Index64.empty(
                starts.length + 1, nplike=self._backend.index_nplike
            )
            assert outoffsets.nplike is self._backend.nplike
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

            tmp = ak.contents.IndexedOptionArray.simplified(
                outindex, out_content, parameters=None
            )
            return ak.contents.ListOffsetArray(outoffsets, tmp, parameters=None)

    def _validity_error(self, path):
        if self._backend.nplike.known_data and self._content.length < self.mask.length:
            return f"at {path} ({type(self)!r}): len(content) < len(mask)"
        else:
            return self._content._validity_error(path + ".content")

    def _nbytes_part(self):
        return self.mask._nbytes_part() + self.content._nbytes_part()

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        elif posaxis is not None and posaxis + 1 == depth + 1:
            mask = ak.index.Index8(self.mask_as_bool(valid_when=False))
            index = ak.index.Index64.empty(
                mask.length, nplike=self._backend.index_nplike
            )
            assert (
                index.nplike is self._backend.index_nplike
                and self._mask.nplike is self._backend.index_nplike
            )
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
                    index.dtype.type,
                    self._mask.dtype.type,
                ](
                    index.data,
                    self._mask.data,
                    self._mask.length,
                )
            )
            next = self.project()._pad_none(target, axis, depth, clip)
            return ak.contents.IndexedOptionArray.simplified(
                index, next, parameters=self._parameters
            )
        else:
            return ak.contents.ByteMaskedArray(
                self._mask,
                self._content._pad_none(target, axis, depth, clip),
                self._valid_when,
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
        this_validbytes = self.mask_as_bool(valid_when=True)

        return self._content._to_arrow(
            pyarrow,
            self,
            ak._connect.pyarrow.and_validbytes(validbytes, this_validbytes),
            length,
            options,
        )

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        cp = Cupy.instance()._module

        assert mask is None  # this class has its own mask
        m = cp.packbits(cp.asarray(self._mask), bitorder="little")
        if m.nbytes % 64:
            m = cp.resize(m, ((m.nbytes // 64) + 1) * 64)
        m = cudf.core.buffer.as_buffer(m)
        inner = self._content._to_cudf(cudf, mask=None, length=length)
        inner.set_base_mask(m)
        return inner

    def _to_backend_array(self, allow_missing, backend):
        return self.to_IndexedOptionArray64()._to_backend_array(allow_missing, backend)

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
        if self._backend.nplike.known_data:
            content = self._content[0 : self._mask.length]
        else:
            content = self._content

        if options["return_array"]:
            if options["return_simplified"]:
                make = ByteMaskedArray.simplified
            else:
                make = ByteMaskedArray

            def continuation():
                return make(
                    self._mask,
                    content._recursively_apply(
                        action,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._valid_when,
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
        if self._content.is_record:
            next = self.to_IndexedOptionArray64()

            content = (
                next._content[: self._mask.length].to_packed(True)
                if recursive
                else next._content[: self._mask.length]
            )

            return ak.contents.IndexedOptionArray(
                next._index, content, parameters=next._parameters
            )

        else:
            content = (
                self._content[: self._mask.length].to_packed(True)
                if recursive
                else self._content[: self._mask.length]
            )

            return ByteMaskedArray(
                self._mask, content, self._valid_when, parameters=self._parameters
            )

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True)
        out = self._content._getitem_range(0, mask.size)._to_list(
            behavior, json_conversions
        )

        for i, isvalid in enumerate(mask):
            if not isvalid:
                out[i] = None

        return out

    def _to_backend(self, backend: Backend) -> Self:
        content = self._content.to_backend(backend)
        mask = self._mask.to_nplike(backend.index_nplike)
        return ByteMaskedArray(
            mask, content, valid_when=self._valid_when, parameters=self._parameters
        )

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and self._valid_when == other.valid_when
            and self._mask.is_equal_to(other.mask, index_dtype, numpyarray)
            and self._content._is_equal_to(
                other.content, index_dtype, numpyarray, all_parameters
            )
        )
