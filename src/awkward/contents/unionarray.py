# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# pylint: disable=consider-using-enumerate

from __future__ import annotations

import copy
import ctypes
from collections.abc import Iterable, Mapping, MutableMapping, Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._layout import maybe_posaxis
from awkward._meta.unionmeta import UnionMeta
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.typetracer import OneOf, TypeTracer
from awkward._nplikes.virtual import VirtualArray
from awkward._parameters import parameters_intersect, parameters_union
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
from awkward.forms.unionform import UnionForm
from awkward.index import Index, Index8, Index64

if TYPE_CHECKING:
    from awkward._slicing import SliceItem

np = NumpyMetadata.instance()
numpy = Numpy.instance()
MAX_UNION_CONTENTS = 2**7  # We use int8 tags, 0-127


@final
class UnionArray(UnionMeta[Content], Content):
    """
    UnionArray represents data drawn from an ordered list of `contents`,
    which can have different types, using

    * `tags`: buffer of integers indicating which content each array element draws from.
    * `index`: buffer of integers indicating which element from the content to draw from.

    UnionArrays correspond to Apache Arrow's
    [dense union type](https://arrow.apache.org/docs/format/Columnar.html#dense-union).
    Awkward Array has no direct equivalent for Apache Arrow's
    [sparse union type](https://arrow.apache.org/docs/format/Columnar.html#sparse-union).

    To illustrate how the constructor arguments are interpreted, the following is a
    simplified implementation of `__init__`, `__len__`, and `__getitem__`:

        class UnionArray(Content):
            def __init__(self, tags, index, contents):
                assert isinstance(tags, Index8)
                assert isinstance(index, (Index32, IndexU32, Index64))
                assert isinstance(contents, list)
                assert len(index) >= len(tags)  # usually equal
                for x in tags:
                    assert 0 <= x < len(contents)
                for i, x in enumerate(tags):
                    assert 0 <= index[i] < len(contents[x])
                self.tags = tags
                self.index = index
                self.contents = contents

            def __len__(self):
                return len(self.tags)

            def __getitem__(self, where):
                if isinstance(where, int):
                    if where < 0:
                        where += len(self)
                    assert 0 <= where < len(self)
                    return self.contents[self.tags[where]][self.index[where]]

                elif isinstance(where, slice) and where.step is None:
                    return UnionArray(
                        self.tags[where], self.index[where], self.contents
                    )

                elif isinstance(where, str):
                    return UnionArray(
                        self.tags, self.index, [x[where] for x in self.contents]
                    )

                else:
                    raise AssertionError(where)
    """

    def __init__(self, tags, index, contents, *, parameters=None):
        if not (isinstance(tags, Index) and tags.dtype == np.dtype(np.int8)):
            raise TypeError(
                f"{type(self).__name__} 'tags' must be an Index with dtype=int8, not {tags!r}"
            )

        if not isinstance(index, Index) and index.dtype in (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
        ):
            raise TypeError(
                f"{type(self).__name__} 'index' must be an Index with dtype in (int32, uint32, int64), "
                f"not {index!r}"
            )

        if not isinstance(contents, Iterable):
            raise TypeError(
                f"{type(self).__name__} 'contents' must be iterable, not {contents!r}"
            )
        if not isinstance(contents, list):
            contents = list(contents)

        if len(contents) < 2:
            raise TypeError(f"{type(self).__name__} must have at least 2 'contents'")

        n_seen_options = 0
        for content in contents:
            if not isinstance(content, Content):
                raise TypeError(
                    f"{type(self).__name__} all 'contents' must be Content subclasses, not {content!r}"
                )
            if content.is_union:
                raise TypeError(
                    "{0} cannot contain union-types in its 'contents' ({1}); try {0}.simplified instead".format(
                        type(self).__name__, type(content).__name__
                    )
                )
            elif content.is_option:
                n_seen_options += 1
            elif content.is_indexed and content.parameter("__array__") != "categorical":
                raise TypeError(
                    f"{type(self).__name__} cannot contain non-categorical indexed-types in its 'contents' ({type(content).__name__}); "
                    f"try {type(self).__name__}.simplified instead"
                )

        if n_seen_options not in {0, len(contents)}:
            raise TypeError(
                f"{type(self).__name__} must either be comprised of entirely optional contents, or no optional contents; "
                f"try {type(self).__name__}.simplified instead"
            )

        backend = None
        for content in contents:
            if backend is None:
                backend = content.backend
            elif backend is not content.backend:
                raise TypeError(
                    f"{type(self).__name__} 'contents' must use the same array library (backend): {type(backend).__name__} vs {type(content.backend).__name__}"
                )

        if (
            backend.index_nplike.known_data
            and tags.length is not unknown_length
            and index.length is not unknown_length
            and tags.length > index.length
        ):
            raise ValueError(
                f"{type(self).__name__} len(tags) ({tags.length}) must be <= len(index) ({index.length})"
            )

        assert tags.nplike is backend.index_nplike
        assert index.nplike is backend.index_nplike

        self._tags = tags
        self._index = index
        self._contents = contents
        self._init(parameters, backend)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    form_cls: Final = UnionForm

    def copy(
        self,
        tags=UNSET,
        index=UNSET,
        contents=UNSET,
        *,
        parameters=UNSET,
    ):
        return UnionArray(
            self._tags if tags is UNSET else tags,
            self._index if index is UNSET else index,
            self._contents if contents is UNSET else contents,
            parameters=self._parameters if parameters is UNSET else parameters,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            tags=copy.deepcopy(self._tags, memo),
            index=copy.deepcopy(self._index, memo),
            contents=[copy.deepcopy(x, memo) for x in self._contents],
            parameters=copy.deepcopy(self._parameters, memo),
        )

    @classmethod
    def simplified(
        cls,
        tags,
        index,
        contents,
        *,
        parameters=None,
        mergebool=False,
    ):
        # Note: to help merge more than 128 arrays, tags *can* have type ak.index.Index64.
        # This is only supported when index is also Index64,
        # and all indexed contents are also Index64.
        # We still require that this reduces to no more than 128 variants.
        self_index = index
        self_tags = tags
        self_contents = contents

        backend = None
        for content in contents:
            if backend is None:
                backend = content.backend
            elif backend is not content.backend:
                raise TypeError(
                    f"{cls.__name__} 'contents' must use the same array library (backend): {type(backend).__name__} vs {type(content.backend).__name__}"
                )

        if backend.nplike.known_data and self_index.length < self_tags.length:
            raise ValueError("invalid UnionArray: len(index) < len(tags)")

        length = self_tags.length
        tags = ak.index.Index8.empty(length, backend.index_nplike)
        index = ak.index.Index64.empty(length, backend.index_nplike)
        contents = []

        # For each outer union content
        for i, self_cont in enumerate(self_contents):
            # Is one of our new contents also a union?
            if isinstance(self_cont, UnionArray):
                innertags = self_cont._tags
                innerindex = self_cont._index
                innercontents = self_cont._contents

                # Update outermost parameters with this union's parameters
                parameters = parameters_union(self_cont._parameters, parameters)

                # For each inner union content
                for j, inner_cont in enumerate(innercontents):
                    unmerged = True

                    # For each "final" outer union content
                    for k in range(len(contents)):
                        # Try and merge inner union content with running outer-union contentca
                        if contents[k]._mergeable_next(inner_cont, mergebool):
                            backend.maybe_kernel_error(
                                backend[
                                    "awkward_UnionArray_simplify",
                                    tags.dtype.type,
                                    index.dtype.type,
                                    self_tags.dtype.type,
                                    self_index.dtype.type,
                                    innertags.dtype.type,
                                    innerindex.dtype.type,
                                ](
                                    tags.data,
                                    index.data,
                                    self_tags.data,
                                    self_index.data,
                                    innertags.data,
                                    innerindex.data,
                                    k,
                                    j,
                                    i,
                                    length,
                                    contents[k].length,
                                )
                            )
                            contents[k] = contents[k]._mergemany([inner_cont])
                            unmerged = False
                            break

                    # Did we fail to merge any of the final outer contents with this inner union content?
                    if unmerged:
                        if len(contents) >= MAX_UNION_CONTENTS:
                            raise ValueError(
                                "UnionArray does not support more than 128 content types"
                            )
                        backend.maybe_kernel_error(
                            backend[
                                "awkward_UnionArray_simplify",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                                innertags.dtype.type,
                                innerindex.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
                                innertags.data,
                                innerindex.data,
                                len(contents),
                                j,
                                i,
                                length,
                                0,
                            )
                        )
                        contents.append(inner_cont)

            else:
                unmerged = True
                for k in range(len(contents)):
                    if contents[k] is self_cont:
                        backend.maybe_kernel_error(
                            backend[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
                                k,
                                i,
                                length,
                                0,
                            )
                        )
                        unmerged = False
                        break

                    elif contents[k]._mergeable_next(self_cont, mergebool):
                        backend.maybe_kernel_error(
                            backend[
                                "awkward_UnionArray_simplify_one",
                                tags.dtype.type,
                                index.dtype.type,
                                self_tags.dtype.type,
                                self_index.dtype.type,
                            ](
                                tags.data,
                                index.data,
                                self_tags.data,
                                self_index.data,
                                k,
                                i,
                                length,
                                contents[k].length,
                            )
                        )
                        contents[k] = contents[k]._mergemany([self_cont])
                        unmerged = False
                        break

                if unmerged:
                    if len(contents) >= MAX_UNION_CONTENTS:
                        raise ValueError(
                            "UnionArray does not support more than 128 content types"
                        )
                    backend.maybe_kernel_error(
                        backend[
                            "awkward_UnionArray_simplify_one",
                            tags.dtype.type,
                            index.dtype.type,
                            self_tags.dtype.type,
                            self_index.dtype.type,
                        ](
                            tags.data,
                            index.data,
                            self_tags.data,
                            self_index.data,
                            len(contents),
                            i,
                            length,
                            0,
                        )
                    )
                    contents.append(self_cont)

        # If any contents are non-categorical index types, we can merge them into the union
        # This is safe, because any remaining index types at this point in the routine are not considered
        # mergeable with the other contents. This means none of the other contents are option or index types,
        # nor are any contents mergeable with these index types' contents. We already know that the
        # other contents cannot be unions. Thus, it's safe to simply erase the outer indexed type.
        for tag, content in enumerate(contents):
            if (
                isinstance(content, ak.contents.IndexedArray)
                and content.parameter("__array__") != "categorical"
            ):
                # only want to affect the part of the UnionArray.index for this tag
                selection = tags.data == tag
                # function-composition of this part of the UnionArray.index with the IndexedArray.index
                index.data[selection] = content.index.data[index.data[selection]]
                # now we don't have an IndexedArray anymore, but we want to preserve its parameters
                contents[tag] = content.content.copy(
                    parameters=parameters_union(
                        content.content.parameters, content.parameters
                    )
                )

        # If any contents are options, ensure all contents are options!
        if any(c.is_option for c in contents):
            contents = [
                c
                if c.is_option
                else c.to_IndexedOptionArray64()
                if c.is_indexed
                else ak.contents.UnmaskedArray(c)
                for c in contents
            ]

        if len(contents) == 1:
            next = contents[0]._carry(index, True)
            return next.copy(parameters=parameters_union(next._parameters, parameters))

        else:
            return cls(
                tags,
                index,
                contents,
                parameters=parameters,
            )

    def _form_with_key(self, getkey: Callable[[Content], str | None]) -> UnionForm:
        form_key = getkey(self)
        return self.form_cls(
            self._tags.form,
            self._index.form,
            [x._form_with_key(getkey) for x in self._contents],
            parameters=self._parameters,
            form_key=form_key,
        )

    def _form_with_key_path(self, path: FormKeyPathT) -> UnionForm:
        return self.form_cls(
            self._tags.form,
            self._index.form,
            [x._form_with_key_path((*path, i)) for i, x in enumerate(self._contents)],
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
        key1 = getkey(self, form, "tags")
        key2 = getkey(self, form, "index")
        container[key1] = ak._util.native_to_byteorder(
            self._tags.raw(backend.index_nplike), byteorder
        )
        container[key2] = ak._util.native_to_byteorder(
            self._index.raw(backend.index_nplike), byteorder
        )
        for i, content in enumerate(self._contents):
            content._to_buffers(form.content(i), getkey, container, backend, byteorder)

    def _to_typetracer(self, forget_length: bool) -> Self:
        tt = TypeTracer.instance()
        tags = self._tags.to_nplike(tt)
        return UnionArray(
            tags.forget_length() if forget_length else tags,
            self._index.to_nplike(tt),
            [x._to_typetracer(forget_length) for x in self._contents],
            parameters=self._parameters,
        )

    def _touch_data(self, recursive: bool):
        self._tags._touch_data()
        self._index._touch_data()
        if recursive:
            for x in self._contents:
                x._touch_data(recursive)

    def _touch_shape(self, recursive: bool):
        self._tags._touch_shape()
        self._index._touch_shape()
        if recursive:
            for x in self._contents:
                x._touch_shape(recursive)

    @property
    def length(self) -> ShapeItem:
        return self._tags.length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnionArray len="]
        out.append(repr(str(self.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._tags._repr(indent + "    ", "<tags>", "</tags>\n"))
        out.append(self._index._repr(indent + "    ", "<index>", "</index>\n"))

        for i, x in enumerate(self._contents):
            out.append(f"{indent}    <content index={str(i)!r}>\n")
            out.append(x._repr(indent + "        ", "", "\n"))
            out.append(f"{indent}    </content>\n")

        out.append(indent + "</UnionArray>")
        out.append(post)
        return "".join(out)

    def _getitem_nothing(self):
        return self._getitem_range(0, 0)

    def _is_getitem_at_placeholder(self) -> bool:
        is_placeholder_tags = isinstance(self._tags, PlaceholderArray)
        is_placeholder_index = isinstance(self._index, PlaceholderArray)
        is_placeholder = is_placeholder_tags or is_placeholder_index
        if is_placeholder:
            return True
        for content in self._contents:
            if content._is_getitem_at_placeholder():
                return True
        return False

    def _is_getitem_at_virtual(self) -> bool:
        is_virtual_tags = (
            isinstance(self._tags, VirtualArray) and not self._tags.is_materialized
        )
        is_virtual_index = (
            isinstance(self._index, VirtualArray) and not self._index.is_materialized
        )
        is_virtual = is_virtual_tags or is_virtual_index
        if is_virtual:
            return True
        for content in self._contents:
            if content._is_getitem_at_virtual():
                return True
        return False

    def _getitem_at(self, where: IndexType):
        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            return OneOf([x._getitem_at(where) for x in self._contents])

        if where < 0:
            where += self.length
        if self._backend.nplike.known_data and not 0 <= where < self.length:
            raise ak._errors.index_error(self, where)
        tag, index = self._tags[where], self._index[where]
        return self._contents[tag]._getitem_at(index)

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        if not self._backend.nplike.known_data:
            self._touch_shape(recursive=False)
            return self

        return UnionArray(
            self._tags[start:stop],
            self._index[start:stop],
            self._contents,
            parameters=self._parameters,
        )

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        return UnionArray.simplified(
            self._tags,
            self._index,
            [x._getitem_field(where, only_fields) for x in self._contents],
            parameters=None,
        )

    def _getitem_fields(
        self, where: list[str | SupportsIndex], only_fields: tuple[str, ...] = ()
    ) -> Content:
        return UnionArray.simplified(
            self._tags,
            self._index,
            [x._getitem_fields(where, only_fields) for x in self._contents],
            parameters=None,
        )

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        assert isinstance(carry, ak.index.Index)

        try:
            nexttags = self._tags[carry.data]
            nextindex = self._index[: self._tags.length][carry.data]
        except IndexError as err:
            raise ak._errors.index_error(self, carry.data, str(err)) from err

        return UnionArray(
            nexttags,
            nextindex,
            self._contents,
            parameters=self._parameters,
        )

    def _union_of_optionarrays(self, index, parameters):
        tag_for_missing = 0
        for i, content in enumerate(self._contents):
            if content.is_option:
                tag_for_missing = i
                break

        if not self._backend.nplike.known_data:
            self._touch_data(recursive=False)
            nexttags = self._tags.data
            nextindex = self._index.data
            contents = []
            for tag, content in enumerate(self._contents):
                if tag == tag_for_missing:
                    indexedoption_index = self._backend.index_nplike.arange(
                        content.length + 1, dtype=np.int64
                    )
                    contents.append(
                        ak.contents.IndexedOptionArray.simplified(
                            ak.index.Index64(indexedoption_index), content
                        )
                    )
                else:
                    contents.append(ak.contents.UnmaskedArray.simplified(content))

        else:
            # like _carry, above
            carry_data = index.raw(self._backend.nplike).copy()
            is_missing = carry_data < 0

            if self._tags.length != 0:
                # but the missing values will temporarily use 0 as a placeholder
                carry_data[is_missing] = 0
                try:
                    nexttags = self._tags.data[carry_data]
                    nextindex = self._index.data[: self._tags.length][carry_data]
                except IndexError as err:
                    raise ak._errors.index_error(self, carry_data, str(err)) from err

                # now actually set the missing values
                nexttags[is_missing] = tag_for_missing
                nextindex[is_missing] = self._contents[tag_for_missing].length

            else:
                # UnionArray is empty, so
                nexttags = self._backend.index_nplike.full(
                    len(carry_data), tag_for_missing, dtype=self._tags.dtype
                )
                nextindex = self._backend.index_nplike.full(
                    len(carry_data),
                    self._contents[tag_for_missing].length,
                    dtype=self._index.dtype,
                )

            contents = []
            for tag, content in enumerate(self._contents):
                if tag == tag_for_missing:
                    indexedoption_index = self._backend.index_nplike.arange(
                        content.length + 1, dtype=np.int64
                    )
                    indexedoption_index[content.length] = -1
                    contents.append(
                        ak.contents.IndexedOptionArray.simplified(
                            ak.index.Index64(indexedoption_index), content
                        )
                    )
                else:
                    contents.append(ak.contents.UnmaskedArray.simplified(content))

        return UnionArray.simplified(
            ak.index.Index(nexttags),
            ak.index.Index(nextindex),
            contents,
            parameters=parameters_union(self._parameters, parameters),
        )

    def project(self, index):
        lentags = self._tags.length
        assert (
            self._index.length is unknown_length or lentags is unknown_length
        ) or self._index.length >= lentags
        lenout = ak.index.Index64.empty(1, self._backend.index_nplike)
        tmpcarry = ak.index.Index64.empty(lentags, self._backend.index_nplike)
        assert (
            lenout.nplike is self._backend.index_nplike
            and tmpcarry.nplike is self._backend.index_nplike
            and self._tags.nplike is self._backend.index_nplike
            and self._index.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_UnionArray_project",
                lenout.dtype.type,
                tmpcarry.dtype.type,
                self._tags.dtype.type,
                self._index.dtype.type,
            ](
                lenout.data,
                tmpcarry.data,
                self._tags.data,
                self._index.data,
                lentags,
                index,
            )
        )
        nextcarry = ak.index.Index64(
            tmpcarry.data[: lenout[0]], nplike=self._backend.index_nplike
        )
        return self._contents[index]._carry(nextcarry, True)

    @staticmethod
    def regular_index(
        tags: Index,
        *,
        backend: Backend,
        index_cls: type[Index] = Index64,
    ):
        tags = tags.to_nplike(backend.index_nplike)

        lentags = tags.length
        _size = ak.index.Index64.empty(1, nplike=backend.index_nplike)
        assert (
            _size.nplike is backend.index_nplike and tags.nplike is backend.index_nplike
        )
        backend.maybe_kernel_error(
            backend[
                "awkward_UnionArray_regular_index_getsize",
                _size.dtype.type,
                tags.dtype.type,
            ](
                _size.data,
                tags.data,
                lentags,
            )
        )
        size = backend.index_nplike.index_as_shape_item(_size[0])
        current = index_cls.empty(size, nplike=backend.index_nplike)
        outindex = index_cls.empty(lentags, nplike=backend.index_nplike)
        assert (
            outindex.nplike is backend.index_nplike
            and current.nplike is backend.index_nplike
            and tags.nplike is backend.index_nplike
        )
        backend.maybe_kernel_error(
            backend[
                "awkward_UnionArray_regular_index",
                outindex.dtype.type,
                current.dtype.type,
                tags.dtype.type,
            ](
                outindex.data,
                current.data,
                size,
                tags.data,
                lentags,
            )
        )
        return outindex

    def _regular_index(self, tags: Index) -> Index:
        return self.regular_index(
            tags, index_cls=type(self._index), backend=self._backend
        )

    @staticmethod
    def nested_tags_index(
        offsets: Index,
        counts: Sequence[Index],
        *,
        backend: Backend,
        tags_cls: type[Index] = Index8,
        index_cls: type[Index] = Index64,
    ) -> tuple[Index, Index]:
        index_nplike = backend.index_nplike

        offsets = offsets.to_nplike(index_nplike)
        counts = [c.to_nplike(index_nplike) for c in counts]

        f_offsets = ak.index.Index64(index_nplike.asarray(offsets.data, copy=True))
        contentlen = index_nplike.index_as_shape_item(
            f_offsets[index_nplike.shape_item_as_index(f_offsets.length - 1)]
        )

        tags = tags_cls.empty(contentlen, nplike=index_nplike)
        index = index_cls.empty(contentlen, nplike=index_nplike)

        for tag, count in enumerate(counts):
            assert (
                tags.nplike is index_nplike
                and index.nplike is index_nplike
                and f_offsets.nplike is index_nplike
                and count.nplike is index_nplike
            )
            backend.maybe_kernel_error(
                backend[
                    "awkward_UnionArray_nestedfill_tags_index",
                    tags.dtype.type,
                    index.dtype.type,
                    f_offsets.dtype.type,
                    count.dtype.type,
                ](
                    tags.data,
                    index.data,
                    f_offsets.data,
                    tag,
                    count.data,
                    f_offsets.length - 1,
                )
            )
        return (tags, index)

    def _getitem_next_jagged_generic(self, slicestarts, slicestops, slicecontent, tail):
        if isinstance(self, ak.contents.UnionArray):
            raise ak._errors.index_error(
                self,
                ak.contents.ListArray(
                    slicestarts, slicestops, slicecontent, parameters=None
                ),
                "cannot apply jagged slices to irreducible union arrays",
            )
        return self._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

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
            outcontents = []
            for i in range(len(self._contents)):
                projection = self.project(i)
                outcontents.append(projection._getitem_next(head, tail, advanced))
            outindex = self._regular_index(self._tags)

            return UnionArray.simplified(
                self._tags,
                outindex,
                outcontents,
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

        elif isinstance(head, ak.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        posaxis = maybe_posaxis(self, axis, depth)

        if posaxis is not None and posaxis + 1 == depth:
            raise AxisError("axis=0 not allowed for flatten")

        else:
            has_offsets = False
            offsetsraws = self._backend.index_nplike.empty(
                len(self._contents), dtype=np.intp
            )
            contents = []

            keep_offsets = []
            for i in range(len(self._contents)):
                offsets, flattened = self._contents[i]._offsets_and_flattened(
                    axis, depth
                )
                offsetsraws[i] = offsets.ptr
                keep_offsets.append(offsets)
                contents.append(flattened)
                has_offsets = offsets.length != 0

            if has_offsets:
                total_length = ak.index.Index64.empty(
                    1, nplike=self._backend.index_nplike
                )
                assert (
                    total_length.nplike is self._backend.index_nplike
                    and self._tags.nplike is self._backend.index_nplike
                    and self._index.nplike is self._backend.index_nplike
                )
                if self._backend.nplike == Numpy.instance():
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_UnionArray_flatten_length",
                            total_length.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                            np.int64,
                        ](
                            total_length.data,
                            self._tags.data,
                            self._index.data,
                            self._tags.length,
                            offsetsraws.ctypes.data_as(
                                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                            ),
                        )
                    )
                elif self._backend.nplike == Cupy.instance():
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_UnionArray_flatten_length",
                            total_length.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                            np.int64,
                        ](
                            total_length.data,
                            self._tags.data,
                            self._index.data,
                            self._tags.length,
                            offsetsraws,
                        )
                    )

                totags = ak.index.Index8.empty(
                    total_length[0], nplike=self._backend.index_nplike
                )
                toindex = ak.index.Index64.empty(
                    total_length[0], nplike=self._backend.index_nplike
                )
                tooffsets = ak.index.Index64.empty(
                    self._tags.length + 1, nplike=self._backend.index_nplike
                )

                assert (
                    totags.nplike is self._backend.index_nplike
                    and toindex.nplike is self._backend.index_nplike
                    and tooffsets.nplike is self._backend.index_nplike
                    and self._tags.nplike is self._backend.index_nplike
                    and self._index.nplike is self._backend.index_nplike
                )
                if self._backend.nplike == Numpy.instance():
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_UnionArray_flatten_combine",
                            totags.dtype.type,
                            toindex.dtype.type,
                            tooffsets.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                            np.int64,
                        ](
                            totags.data,
                            toindex.data,
                            tooffsets.data,
                            self._tags.data,
                            self._index.data,
                            self._tags.length,
                            offsetsraws.ctypes.data_as(
                                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))
                            ),
                        )
                    )
                elif self._backend.nplike == Cupy.instance():
                    self._backend.maybe_kernel_error(
                        self._backend[
                            "awkward_UnionArray_flatten_combine",
                            totags.dtype.type,
                            toindex.dtype.type,
                            tooffsets.dtype.type,
                            self._tags.dtype.type,
                            self._index.dtype.type,
                            np.int64,
                        ](
                            totags.data,
                            toindex.data,
                            tooffsets.data,
                            self._tags.data,
                            self._index.data,
                            self._tags.length,
                            offsetsraws,
                        )
                    )
                return (
                    tooffsets,
                    UnionArray(
                        totags,
                        toindex,
                        contents,
                        parameters=self._parameters,
                    ),
                )

            else:
                offsets = ak.index.Index64.zeros(
                    0,
                    nplike=self._backend.index_nplike,
                    dtype=np.int64,
                )
                return (
                    offsets,
                    UnionArray(
                        self._tags,
                        self._index,
                        contents,
                        parameters=self._parameters,
                    ),
                )

    def _mergeable_next(self, other: Content, mergebool: bool) -> bool:
        return True

    def _merging_strategy(self, others):
        if len(others) == 0:
            raise ValueError(
                "to merge this array with 'others', at least one other must be provided"
            )

        head = [self, *others]
        tail = []

        if any(x.backend.nplike.known_data for x in head + tail) and not all(
            x.backend.nplike.known_data for x in head + tail
        ):
            raise RuntimeError

        return head, tail

    def _reverse_merge(self, other):
        theirlength = other.length
        mylength = self.length

        tags = ak.index.Index8.empty(
            theirlength + mylength,
            nplike=self._backend.index_nplike,
        )
        index = ak.index.Index64.empty(
            theirlength + mylength,
            nplike=self._backend.index_nplike,
        )

        contents = [other]
        contents.extend(self.contents)

        assert tags.nplike is self._backend.index_nplike
        self._backend.maybe_kernel_error(
            self._backend["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data,
                0,
                theirlength,
                0,
            )
        )

        assert index.nplike is self._backend.index_nplike
        self._backend.maybe_kernel_error(
            self._backend["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data,
                0,
                theirlength,
            )
        )

        assert (
            tags.nplike is self._backend.index_nplike
            and self.tags.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_UnionArray_filltags",
                tags.dtype.type,
                self.tags.dtype.type,
            ](
                tags.data,
                theirlength,
                self.tags.data,
                mylength,
                1,
            )
        )

        assert (
            index.nplike is self._backend.index_nplike
            and self.index.nplike is self._backend.index_nplike
        )
        self._backend.maybe_kernel_error(
            self._backend[
                "awkward_UnionArray_fillindex",
                index.dtype.type,
                self.index.dtype.type,
            ](
                index.data,
                theirlength,
                self.index.data,
                mylength,
            )
        )

        if len(contents) > MAX_UNION_CONTENTS:
            raise ValueError("UnionArray cannot have more than 128 content types")

        return ak.contents.UnionArray.simplified(
            tags, index, contents, parameters=self._parameters
        )

    def _mergemany(self, others: Sequence[Content]) -> Content:
        if len(others) == 0:
            return self

        index_nplike = self._backend.index_nplike
        head, tail = self._merging_strategy(others)

        total_length = 0
        for array in head:
            total_length += array.length

        nexttags = ak.index.Index8.empty(total_length, nplike=index_nplike)
        nextindex = ak.index.Index64.empty(total_length, nplike=index_nplike)

        nextcontents = []
        length_so_far = 0

        parameters = self._parameters
        for array in head:
            if isinstance(array, ak.contents.EmptyArray):
                continue

            parameters = parameters_intersect(parameters, array._parameters)
            if isinstance(array, ak.contents.UnionArray):
                union_tags = array.tags
                union_index = array.index
                union_contents = array.contents
                assert (
                    nexttags.nplike is index_nplike
                    and union_tags.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_UnionArray_filltags",
                        nexttags.dtype.type,
                        union_tags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        union_tags.data,
                        array.length,
                        len(nextcontents),
                    )
                )
                assert (
                    nextindex.nplike is index_nplike
                    and union_index.nplike is index_nplike
                )
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_UnionArray_fillindex",
                        nextindex.dtype.type,
                        union_index.dtype.type,
                    ](
                        nextindex.data,
                        length_so_far,
                        union_index.data,
                        array.length,
                    )
                )
                length_so_far += array.length

                nextcontents.extend(union_contents)

            # This pathway is covered by `.simplified`, but we can avoid an extra array operation
            # by performing this now
            elif (
                isinstance(array, ak.contents.IndexedArray)
                and array.parameter("__array__") != "categorical"
            ):
                assert nexttags.nplike is index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_UnionArray_filltags_const",
                        nexttags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        array.length,
                        len(nextcontents),
                    )
                )
                nextindex.data[
                    index_nplike.shape_item_as_index(
                        length_so_far
                    ) : index_nplike.shape_item_as_index(length_so_far + array.length)
                ] = array.index.data

                length_so_far += array.length

                nextcontents.append(
                    array.content.copy(
                        parameters=parameters_union(
                            array.content._parameters, array._parameters
                        )
                    )
                )
            else:
                assert nexttags.nplike is index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_UnionArray_filltags_const",
                        nexttags.dtype.type,
                    ](
                        nexttags.data,
                        length_so_far,
                        array.length,
                        len(nextcontents),
                    )
                )

                assert nextindex.nplike is index_nplike
                self._backend.maybe_kernel_error(
                    self._backend[
                        "awkward_UnionArray_fillindex_count", nextindex.dtype.type
                    ](nextindex.data, length_so_far, array.length)
                )

                length_so_far += array.length

                nextcontents.append(array)

        if len(nextcontents) > MAX_UNION_CONTENTS:
            raise ValueError("UnionArray cannot have more than 128 content types")

        next = ak.contents.UnionArray.simplified(
            nexttags,
            nextindex,
            nextcontents,
            parameters=parameters,
        )

        # `tail` should always be empty for UnionArray's merging strategy
        assert len(tail) == 0
        return next

    def _fill_none(self, value: Content) -> Content:
        contents = []
        for content in self._contents:
            contents.append(content._fill_none(value))
        return UnionArray.simplified(
            self._tags,
            self._index,
            contents,
            parameters=self._parameters,
        )

    def _local_index(self, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._local_index_axis0()
        else:
            contents = []
            for content in self._contents:
                contents.append(content._local_index(axis, depth))
            return UnionArray(
                self._tags,
                self._index,
                contents,
                parameters=self._parameters,
            )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            contents = []
            for content in self._contents:
                contents.append(
                    content._combinations(
                        n, replacement, recordlookup, parameters, axis, depth
                    )
                )
            return ak.unionarray.UnionArray(
                self._tags,
                self._index,
                contents,
                parameters=self._parameters,
            )

    def _numbers_to_type(self, name, including_unknown):
        contents = []
        for x in self._contents:
            contents.append(x._numbers_to_type(name, including_unknown))
        return ak.contents.UnionArray(
            self._tags,
            self._index,
            contents,
            parameters=self._parameters,
        )

    def _is_unique(self, negaxis, starts, parents, outlength):
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if isinstance(simplified, ak.contents.UnionArray):
            raise ValueError("cannot check if an irreducible UnionArray is unique")

        return simplified._is_unique(negaxis, starts, parents, outlength)

    def _unique(self, negaxis, starts, parents, outlength):
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if isinstance(simplified, ak.contents.UnionArray):
            raise ValueError("cannot make a unique irreducible UnionArray")

        return simplified._unique(negaxis, starts, parents, outlength)

    def _argsort_next(
        self, negaxis, starts, shifts, parents, outlength, ascending, stable
    ):
        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if simplified.length is not unknown_length and simplified.length == 0:
            return ak.contents.NumpyArray(
                self._backend.nplike.empty(0, dtype=np.int64),
                parameters=None,
                backend=self._backend,
            )

        if isinstance(simplified, ak.contents.UnionArray):
            raise ValueError("cannot argsort an irreducible UnionArray")

        return simplified._argsort_next(
            negaxis, starts, shifts, parents, outlength, ascending, stable
        )

    def _sort_next(self, negaxis, starts, parents, outlength, ascending, stable):
        if self.length is not unknown_length and self.length == 0:
            return self

        simplified = type(self).simplified(
            self._tags,
            self._index,
            self._contents,
            parameters=self._parameters,
            mergebool=True,
        )
        if simplified.length is not unknown_length and simplified.length == 0:
            return simplified

        if isinstance(simplified, ak.contents.UnionArray):
            raise ValueError("cannot sort an irreducible UnionArray")

        return simplified._sort_next(
            negaxis, starts, parents, outlength, ascending, stable
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
        # If we have a UnionArray, it must be irreducible, thanks to the
        # canonical checks in the constructor.
        raise ValueError(f"cannot call ak.{reducer.name} on an irreducible UnionArray")

    def _validity_error(self, path):
        if self._backend.nplike.known_data and self.index.length < self.tags.length:
            return f"at {path} ({type(self)!r}): len(index) < len(tags)"

        lencontents = self._backend.index_nplike.empty(
            len(self.contents), dtype=np.int64
        )
        if self._backend.nplike.known_data:
            for j, _content_j in enumerate(self._contents):
                lencontents[j] = _content_j.length

        error = self._backend[
            "awkward_UnionArray_validity",
            self.tags.dtype.type,
            self.index.dtype.type,
            np.int64,
        ](
            self.tags.data,
            self.index.data,
            self.tags.length,
            len(self.contents),
            lencontents,
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

        # Check triangular pairs (i, j) are not mergeable
        for i, content_i in enumerate(self._contents):
            for j in range(i):
                content_j = self._contents[j]
                if ak._do.mergeable(content_i, content_j, mergebool=False):
                    return f"at {path}: content({i}) is mergeable with content({j})"

        for i, content in enumerate(self._contents):
            sub = content._validity_error(path + f".content({i})")
            if sub != "":
                return sub

        return ""

    def _nbytes_part(self):
        result = self.tags._nbytes_part() + self.index._nbytes_part()
        for content in self.contents:
            result = result + content._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        posaxis = maybe_posaxis(self, axis, depth)
        if posaxis is not None and posaxis + 1 == depth:
            return self._pad_none_axis0(target, clip)
        else:
            contents = []
            for content in self._contents:
                contents.append(content._pad_none(target, axis, depth, clip))
            return ak.contents.UnionArray.simplified(
                self.tags,
                self.index,
                contents,
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
        nptags = self._tags.raw(numpy)
        npindex = self._index.raw(numpy)
        copied_index = False

        values = []
        for tag, content in enumerate(self._contents):
            selected_tags = nptags == tag
            this_index = npindex[selected_tags]

            # Arrow unions can't have masks; propagate validbytes down to the content.
            if validbytes is not None:
                # If this_index is a filtered permutation, we can just filter-permute
                # the mask to have the same order the content.
                if numpy.unique_values(this_index).shape[0] == this_index.shape[0]:
                    this_validbytes = numpy.zeros(this_index.shape[0], dtype=np.int8)
                    this_validbytes[this_index] = validbytes[selected_tags]

                # If this_index is not a filtered permutation, then we can't modify
                # the mask to fit the content. The same element in the content array
                # will appear multiple times in the union array, and it *might* be
                # presented as valid in some union array elements and masked in others.
                # The validbytes that recurses down to the next level can't have an
                # element that is both 0 (masked) and 1 (valid).
                else:
                    this_validbytes = validbytes[selected_tags]
                    content = content[this_index]
                    if not copied_index:
                        copied_index = True
                        npindex = numpy.asarray(npindex, copy=True)
                    npindex[selected_tags] = numpy.arange(
                        this_index.shape[0], dtype=npindex.dtype
                    )

            else:
                this_validbytes = None

            this_length = 0
            if len(this_index) != 0:
                this_length = this_index.max() + 1

            values.append(
                content._to_arrow(
                    pyarrow, mask_node, this_validbytes, this_length, options
                )
            )

        types = pyarrow.union(
            [
                pyarrow.field(str(i), values[i].type).with_nullable(
                    mask_node is not None
                    or self._contents[i]._arrow_needs_option_type()
                )
                for i in range(len(values))
            ],
            "dense",
            list(range(len(values))),
        )

        if not issubclass(npindex.dtype.type, np.int32):
            npindex = numpy.astype(npindex, dtype=np.int32)

        return pyarrow.Array.from_buffers(
            ak._connect.pyarrow.to_awkwardarrow_type(
                types,
                options["extensionarray"],
                options["record_is_scalar"],
                None,
                self,
            ),
            nptags.shape[0],
            [
                None,
                ak._connect.pyarrow.to_length(nptags, length),
                ak._connect.pyarrow.to_length(npindex, length),
            ],
            children=values,
        )

    def _to_backend_array(self, allow_missing, backend):
        raise TypeError(
            "Conversion of irreducible unions to backend arrays is not supported."
        )

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        out = []
        for i in range(len(self._contents)):
            index = self._index[self._tags.data == i]
            out.extend(
                self._contents[i]
                ._carry(index, False)
                ._remove_structure(backend, options)
            )
        return out

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        if options["return_array"]:
            if options["return_simplified"]:
                make = UnionArray.simplified
            else:
                make = UnionArray

            def continuation():
                return make(
                    self._tags,
                    self._index,
                    [
                        content._recursively_apply(
                            action,
                            depth,
                            copy.copy(depth_context),
                            lateral_context,
                            options,
                        )
                        for content in self._contents
                    ],
                    parameters=self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                for content in self._contents:
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
        index_nplike = self._backend.index_nplike
        tags = self._tags.data
        original_index = index = self._index.data[: tags.shape[0]]

        contents = list(self._contents)

        for tag in range(len(self._contents)):
            is_tag = tags == tag
            num_tag = index_nplike.index_as_shape_item(
                index_nplike.count_nonzero(is_tag)
            )

            if len(contents[tag]) > num_tag:
                if original_index is index:
                    index = index.copy()
                index[is_tag] = self._backend.index_nplike.arange(
                    num_tag, dtype=index.dtype
                )
                contents[tag] = self.project(tag)

            contents[tag] = (
                contents[tag].to_packed(True) if recursive else contents[tag]
            )

        return UnionArray(
            ak.index.Index8(tags, nplike=self._backend.index_nplike),
            ak.index.Index(index, nplike=self._backend.index_nplike),
            contents,
            parameters=self._parameters,
        )

    def _to_list(self, behavior, json_conversions):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot convert typetracer arrays to Python lists")

        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        tags = self._tags.raw(numpy)
        index = self._index.raw(numpy)
        contents = [x._to_list(behavior, json_conversions) for x in self._contents]

        out = [None] * tags.shape[0]
        for i, tag in enumerate(tags):
            out[i] = contents[tag][index[i]]
        return out

    def _to_backend(self, backend: Backend) -> Self:
        tags = self._tags.to_nplike(backend.index_nplike)
        index = self._index.to_nplike(backend.index_nplike)
        contents = [content.to_backend(backend) for content in self._contents]
        return UnionArray(
            tags,
            index,
            contents,
            parameters=self._parameters,
        )

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        return (
            self._is_equal_to_generic(other, all_parameters)
            and self._tags.is_equal_to(other.tags)
            and self._index.is_equal_to(other.index, index_dtype, numpyarray)
            and len(self._contents) == len(other.contents)
            and all(
                content.is_equal_to(other.contents[i], index_dtype, numpyarray)
                for i, content in enumerate(self.contents)
            )
        )
