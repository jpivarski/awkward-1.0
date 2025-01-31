# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence, Sized
from numbers import Complex, Real

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import (
    backend_of,
    register_backend_lookup_factory,
    regularize_backend,
)
from awkward._behavior import get_array_class, get_record_class
from awkward._kernels import KernelError
from awkward._layout import maybe_posaxis, wrap_layout
from awkward._meta.meta import Meta
from awkward._namedaxis import (
    NamedAxis,
    _add_named_axis,
    _keep_named_axis,
    _remove_named_axis,
)
from awkward._nplikes import to_nplike
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import IndexType, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import (
    parameters_are_equal,
    type_parameters_equal,
)
from awkward._regularize import (
    is_array_like,
    is_integer,
    is_integer_like,
    is_sized_iterable,
)
from awkward._slicing import normalize_slice
from awkward._typing import (
    TYPE_CHECKING,
    Any,
    AxisMaybeNone,
    JSONMapping,
    Literal,
    Protocol,
    Self,
    SupportsIndex,
    Type,
    TypeAlias,
    TypedDict,
)
from awkward._util import UNSET
from awkward.forms.form import Form, FormKeyPathT
from awkward.index import Index, Index64

if TYPE_CHECKING:
    from awkward._nplikes.numpy import NumpyLike
    from awkward._slicing import SliceItem
    from awkward.contents.indexedoptionarray import IndexedOptionArray
    from awkward.contents.numpyarray import NumpyArray
    from awkward.contents.regulararray import RegularArray


np = NumpyMetadata.instance()
numpy = Numpy.instance()

ActionType: TypeAlias = """Callable[
    [
        Content,
        int,
        dict | None,
        dict | None,
        Callable[[], None],
        dict | None,
        NumpyLike | None,
        dict[str, Any],
    ],
    Content | None,
]"""
JSONValueType: TypeAlias = """
float | int | str | list[JSONValueType] | dict[str, JSONValueType]
"""


class ImplementsApplyAction(Protocol):
    def __call__(
        self,
        layout: Content,
        *,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        continuation: Callable[[], Content],
        behavior: Mapping | None,
        backend: Backend,
        options: ApplyActionOptions,
    ) -> Content | None: ...


class ApplyActionOptions(TypedDict):
    allow_records: bool
    keep_parameters: bool
    numpy_to_regular: bool
    regular_to_jagged: bool
    return_simplified: bool
    return_array: bool
    function_name: str | None


class RemoveStructureOptions(TypedDict):
    flatten_records: bool
    function_name: str
    drop_nones: bool
    keepdims: bool
    allow_records: bool
    list_to_regular: bool


class ToArrowOptions(TypedDict):
    list_to32: bool
    string_to32: bool
    bytestring_to32: bool
    emptyarray_to: np.dtype | None
    categorical_as_dictionary: bool
    extensionarray: bool
    count_nulls: bool
    record_is_scalar: bool


class Content(Meta):
    def _init(self, parameters: dict[str, Any] | None, backend: Backend):
        if parameters is None:
            pass
        elif not isinstance(parameters, dict):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be a dict or None, not {parameters!r}"
            )
        # Validate built-in `__array__`
        elif parameters.get("__array__") is not None:
            array_name = parameters["__array__"]
            if not self.is_list and array_name in (
                "string",
                "bytestring",
            ):
                raise TypeError(
                    '{} is not allowed to have parameters["__array__"] = "{}"'.format(
                        type(self).__name__, parameters["__array__"]
                    )
                )
            if not isinstance(self, ak.contents.NumpyArray) and parameters.get(
                "__array__"
            ) in ("char", "byte"):
                raise TypeError(
                    '{} is not allowed to have parameters["__array__"] = "{}"'.format(
                        type(self).__name__, parameters["__array__"]
                    )
                )
            if not self.is_indexed and array_name == "categorical":
                raise TypeError(
                    '{} is not allowed to have parameters["__array__"] = "{}"'.format(
                        type(self).__name__, parameters["__array__"]
                    )
                )
            if not self.is_record and array_name == "sorted_map":
                raise TypeError(
                    '{} is not allowed to have parameters["__array__"] = "{}"'.format(
                        type(self).__name__, parameters["__array__"]
                    )
                )
        # TODO: enable this once we can guarantee this doesn't happen during broadcasting
        # elif not (self.is_list or parameters.get("__list__") is None):
        #     raise TypeError(
        #         '{} is not allowed to have parameters["__list__"] = "{}"'.format(
        #             type(self).__name__, parameters["__list__"]
        #         )
        #     )
        # elif not (self.is_record or parameters.get("__record__") is None):
        #     raise TypeError(
        #         '{} is not allowed to have parameters["__record__"] = "{}"'.format(
        #             type(self).__name__, parameters["__record__"]
        #         )
        #     )

        if not isinstance(backend, Backend):
            raise TypeError(
                f"{type(self).__name__} 'backend' must be a Backend, not {backend!r}"
            )

        self._parameters = parameters
        self._backend = backend

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def form(self) -> Form:
        return self.form_with_key(None)

    def form_with_key(
        self, form_key: str | None | Callable = "node{id}", id_start: int = 0
    ) -> Form:
        hold_id = [id_start]

        if form_key is None:

            def getkey(layout):
                return None

        elif isinstance(form_key, str):

            def getkey(layout):
                out = form_key.format(id=hold_id[0])
                hold_id[0] += 1
                return out

        elif callable(form_key):

            def getkey(layout):
                out = form_key(id=hold_id[0], layout=layout)
                hold_id[0] += 1
                return out

        else:
            raise TypeError(
                f"form_key must be None, a string, or a callable, not {type(form_key)}"
            )

        return self._form_with_key(getkey)

    def _form_with_key(
        self,
        getkey: Callable[[Content], str | None],
    ) -> Form:
        raise NotImplementedError

    def form_with_key_path(self, root: FormKeyPathT = ()) -> Form:
        return self._form_with_key_path(root)

    def _form_with_key_path(self, path: FormKeyPathT) -> Form:
        raise NotImplementedError

    @property
    def form_cls(self) -> type[Form]:
        raise NotImplementedError

    def to_typetracer(self, forget_length: bool = False) -> Self:
        return self._to_typetracer(forget_length)

    def _to_typetracer(self, forget_length: bool) -> Self:
        raise NotImplementedError

    def _touch_data(self, recursive: bool):
        raise NotImplementedError

    def _touch_shape(self, recursive: bool):
        raise NotImplementedError

    @property
    def length(self) -> ShapeItem:
        raise NotImplementedError

    def _to_buffers(
        self,
        form: Form,
        getkey: Callable[[Content, Form, str], str],
        container: MutableMapping[str, Any] | None,
        backend: Backend,
        byteorder: Literal["<", ">"],
    ) -> tuple[Form, int, Mapping[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return int(self.length)

    def _repr_extra(self, indent: str) -> list[str]:
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                out.append(f"\n{indent}<parameter name={k!r}>{v!r}</parameter>")
        return out

    def maybe_to_NumpyArray(self):
        return None

    def _maybe_index_error(self, error: KernelError | None, slicer):
        if error is None or error.str is None:
            return
        else:
            message = self._backend.format_kernel_error(error)
            raise ak._errors.index_error(self, slicer, message)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise TypeError(
            "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak.highlevel.Array"
        )

    def __array_function__(self, func, types, args, kwargs):
        raise TypeError(
            "do not apply NumPy functions to low-level layouts (Content subclasses); put them in ak.highlevel.Array"
        )

    def __array__(self, dtype=None):
        raise TypeError(
            "do not try to convert low-level layouts (Content subclasses) into NumPy arrays; put them in ak.highlevel.Array"
        )

    def __iter__(self):
        if not self._backend.nplike.known_data:
            raise TypeError("cannot iterate on an array without concrete data")

        for i in range(len(self)):
            yield self._getitem_at(i)

    def _getitem_next_field(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ):
        nexthead, nexttail = ak._slicing.head_tail(tail)
        return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

    def _getitem_next_fields(
        self, head: SliceItem, tail: tuple[SliceItem, ...], advanced: Index | None
    ) -> Content:
        only_fields, not_fields = [], []
        for x in tail:
            if isinstance(x, (str, list)):
                only_fields.append(x)
            else:
                not_fields.append(x)
        nexthead, nexttail = ak._slicing.head_tail(tuple(not_fields))
        return self._getitem_fields(head, tuple(only_fields))._getitem_next(
            nexthead, nexttail, advanced
        )

    def _getitem_next_newaxis(
        self, tail: tuple[SliceItem, ...], advanced: Index | None
    ) -> RegularArray:
        nexthead, nexttail = ak._slicing.head_tail(tail)
        return ak.contents.RegularArray(
            self._getitem_next(nexthead, nexttail, advanced), 1, 0, parameters=None
        )

    def _getitem_next_ellipsis(
        self, tail: tuple[SliceItem, ...], advanced: Index | None
    ) -> Content:
        mindepth, maxdepth = self.minmax_depth

        dimlength = sum(
            1 if is_integer_like(x) or isinstance(x, (slice, Index64)) else 0
            for x in tail
        )

        if len(tail) == 0 or mindepth - 1 == maxdepth - 1 == dimlength:
            nexthead, nexttail = ak._slicing.head_tail(tail)
            return self._getitem_next(nexthead, nexttail, advanced)

        elif dimlength in {mindepth - 1, maxdepth - 1}:
            raise ak._errors.index_error(
                self,
                Ellipsis,
                "ellipsis (`...`) can't be used on data with different numbers of dimensions",
            )

        else:
            return self._getitem_next(slice(None), (Ellipsis, *tail), advanced)

    def _getitem_next_regular_missing(
        self,
        head: IndexedOptionArray,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
        raw: Content,
        length: int,
    ) -> RegularArray:
        # if this is in a tuple-slice and really should be 0, it will be trimmed later
        length = 1 if length is not unknown_length and length == 0 else length
        index = head.index
        indexlength = index.length
        index = index.to_nplike(self._backend.index_nplike)
        outindex = Index64.empty(
            index.length * length,
            self._backend.index_nplike,
        )

        assert (
            outindex.nplike is self._backend.index_nplike
            and index.nplike is self._backend.index_nplike
        )
        self._maybe_index_error(
            self._backend[
                "awkward_missing_repeat", outindex.dtype.type, index.dtype.type
            ](
                outindex.data,
                index.data,
                index.length,
                length,
                raw._size,
            ),
            slicer=head,
        )

        out = ak.contents.IndexedOptionArray.simplified(
            outindex, raw.content, parameters=self._parameters
        )
        return ak.contents.RegularArray(
            out, indexlength, 1, parameters=self._parameters
        )

    def _getitem_next_missing_jagged(
        self, head: Content, tail, advanced: Index | None, that: Content
    ) -> RegularArray:
        head = head.to_backend(self._backend)
        jagged = head.content.to_ListOffsetArray64()
        index = head._index
        content = that._getitem_at(0)
        if self._backend.nplike.known_data and content.length < index.length:
            raise ak._errors.index_error(
                self,
                head,
                f"cannot fit masked jagged slice with length {index.length} into {type(that).__name__} of size {content.length}",
            )

        outputmask = Index64.empty(index.length, self._backend.index_nplike)
        starts = Index64.empty(index.length, self._backend.index_nplike)
        stops = Index64.empty(index.length, self._backend.index_nplike)

        assert (
            index.nplike is self._backend.index_nplike
            and jagged._offsets.nplike is self._backend.index_nplike
            and outputmask.nplike is self._backend.index_nplike
            and starts.nplike is self._backend.index_nplike
            and stops.nplike is self._backend.index_nplike
        )
        self._maybe_index_error(
            self._backend[
                "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
                index.dtype.type,
                jagged._offsets.dtype.type,
                outputmask.dtype.type,
                starts.dtype.type,
                stops.dtype.type,
            ](
                index.data,
                jagged._offsets.data,
                outputmask.data,
                starts.data,
                stops.data,
                index.length,
            ),
            slicer=head,
        )

        tmp = content._getitem_next_jagged(starts, stops, jagged.content, tail)
        out = ak.contents.IndexedOptionArray.simplified(
            outputmask, tmp, parameters=self._parameters
        )
        return ak.contents.RegularArray(
            out, index.length, 1, parameters=self._parameters
        )

    def _getitem_next_missing(
        self,
        head: IndexedOptionArray,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        assert isinstance(head, ak.contents.IndexedOptionArray)

        if advanced is not None:
            raise ak._errors.index_error(
                self,
                head,
                "cannot mix missing values in slice with NumPy-style advanced indexing",
            )

        if isinstance(head.content, ak.contents.ListOffsetArray):
            if self._backend.nplike.known_data and self.length != 1:
                raise NotImplementedError("reached a not-well-considered code path")
            return self._getitem_next_missing_jagged(head, tail, advanced, self)

        if isinstance(head.content, ak.contents.NumpyArray):
            headcontent = Index64(head.content.data)
            nextcontent = self._getitem_next(headcontent, tail, advanced)
        else:
            nextcontent = self._getitem_next(head.content, tail, advanced)

        if isinstance(nextcontent, ak.contents.RegularArray):
            return self._getitem_next_regular_missing(
                head, tail, advanced, nextcontent, nextcontent.length
            )

        elif isinstance(nextcontent, ak.contents.RecordArray):
            if len(nextcontent._fields) == 0:
                return nextcontent

            contents = []

            for content in nextcontent.contents:
                if isinstance(content, ak.contents.RegularArray):
                    contents.append(
                        self._getitem_next_regular_missing(
                            head, tail, advanced, content, content.length
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"FIXME: unhandled case of SliceMissing with RecordArray containing {content}"
                    )

            return ak.contents.RecordArray(
                contents,
                nextcontent._fields,
                parameters=self._parameters,
                backend=self._backend,
            )

        else:
            raise NotImplementedError(
                f"FIXME: unhandled case of SliceMissing with {nextcontent}"
            )

    def __getitem__(self, where):
        return self._getitem(where, NamedAxis)

    def _getitem(self, where, named_axis: Type[NamedAxis] = NamedAxis):
        if is_integer_like(where):
            # propagate named_axis to output
            named_axis.mapping = _remove_named_axis(
                named_axis.mapping, where, self.purelist_depth
            )
            return self._getitem_at(ak._slicing.normalize_integer_like(where))

        elif isinstance(where, slice) and where.step is None:
            # Ensure that start, stop are non-negative!
            start, stop, _, _ = self._backend.index_nplike.derive_slice_for_length(
                normalize_slice(where, nplike=self._backend.index_nplike), self.length
            )
            return self._getitem_range(start, stop)

        elif isinstance(where, slice):
            return self._getitem((where,), named_axis)

        elif isinstance(where, str):
            return self._getitem_field(where)

        elif where is np.newaxis:
            return self._getitem((where,), named_axis)

        elif where is Ellipsis:
            return self._getitem((where,), named_axis)

        elif isinstance(where, tuple):
            if len(where) == 0:
                return self

            # count number of ellipsis
            # Need to use a little trick here:
            #   where.count(Ellipsis) does not work, because it will do a == comparison against Ellipsis,
            #   and this will fail in the case of typetracers where == is dispatched to np.equal ufunc.
            #   In this dispatch we encounter an assertion that the type of the Ellipsis is not allowed.
            #   ...but luckily we can use the fact that Ellipsis is a singleton and use the 'is' operator
            n_ellipsis = 0
            for w in where:
                if w is ...:
                    n_ellipsis += 1

            if n_ellipsis > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")

            # Backend may change if index contains typetracers
            backend = backend_of(self, *where, coerce_to_common=True)
            this = self.to_backend(backend)

            # Normalise valid indices onto well-defined basis
            items = ak._slicing.normalise_items(where, backend)
            # Prepare items for advanced indexing (e.g. via broadcasting)
            nextwhere = ak._slicing.prepare_advanced_indexing(items, backend)

            # Handle named axis
            # first expand the ellipsis to colons in nextwhere,
            # copy nextwhere to not pollute the original
            _nextwhere = tuple(nextwhere)
            if n_ellipsis == 1:
                # collect the ellipsis index
                # same little trick as above for `nextwhere.index(...)`
                (ellipsis_at,) = tuple(i for i, x in enumerate(nextwhere) if x is ...)
                # calculate how many slice(None) we need to add
                # same little trick as above for `nextwhere.count(None)`
                n_newaxis = 0
                for x in nextwhere:
                    if x is np.newaxis or x is None:
                        n_newaxis += 1
                n_total = self.minmax_depth[1]
                n_slice_none = n_total - (len(_nextwhere) - n_newaxis - 1)
                # expand `[...]` to `[:]*n_slice_none`
                _nextwhere = (
                    _nextwhere[:ellipsis_at]
                    + (slice(None),) * n_slice_none
                    + _nextwhere[ellipsis_at + 1 :]
                )

            # now propagate named axis
            _named_axis = _keep_named_axis(named_axis.mapping, None)
            _adjust_dim = 0
            # this loop does the following:
            # - remove a named axis for integer indices, e.g. `a[1, 2]`
            # - add a named axis for None (or np.newaxis) indices, e.g. `a[..., None]`
            # - keep named axis for any other index, e.g. `a[:]`, `a[0:1]`, or `a[a>0]`
            #   (these may only remove elements, but not dimensions)
            for dim, nw in enumerate(_nextwhere):
                dim_adjusted = dim + _adjust_dim
                total_adjusted = self.minmax_depth[1] + _adjust_dim
                for _, pos in _named_axis.items():
                    if maybe_posaxis(self, pos, 0) == dim_adjusted:
                        break

                if is_integer(nw) or (is_array_like(nw) and nw.ndim == 0):
                    _named_axis = _remove_named_axis(
                        named_axis=_named_axis,
                        axis=dim_adjusted,
                        total=total_adjusted,
                    )
                    _adjust_dim -= 1
                elif nw is None:
                    _named_axis = _add_named_axis(
                        named_axis=_named_axis,
                        axis=dim_adjusted,
                        total=total_adjusted,
                    )
                    _adjust_dim += 1

            # set propagated named axis
            named_axis.mapping = _named_axis

            next = ak.contents.RegularArray(
                this,
                this.length,
                1,
                parameters=None,
            )

            out = next._getitem_next(nextwhere[0], nextwhere[1:], None)

            if out.length is not unknown_length and out.length == 0:
                return out._getitem_nothing()
            else:
                return out._getitem_at(0)

        elif isinstance(where, ak.highlevel.Array):
            return self._getitem(where.layout, named_axis)

        # Convert between nplikes of different backends
        elif (
            isinstance(where, ak.contents.Content)
            and where.backend is not self._backend
        ):
            backend = backend_of(self, where, coerce_to_common=True)
            return self.to_backend(backend)._getitem(
                where.to_backend(backend), named_axis
            )

        elif isinstance(where, ak.contents.NumpyArray):
            data_as_index = to_nplike(
                where.data,
                self._backend.index_nplike,
                from_nplike=self._backend.nplike,
            )
            if np.issubdtype(where.dtype, np.int64):
                allow_lazy = True
                carry = Index64(
                    self._backend.index_nplike.reshape(data_as_index, (-1,)),
                    nplike=self._backend.index_nplike,
                )
            elif np.issubdtype(where.dtype, np.integer):
                allow_lazy = "copied"  # True, but also can be modified in-place
                carry = Index64(
                    self._backend.index_nplike.reshape(
                        self._backend.index_nplike.astype(
                            data_as_index, dtype=np.int64, copy=True
                        ),
                        (-1,),
                    ),
                    nplike=self._backend.index_nplike,
                )
            elif np.issubdtype(where.dtype, np.bool_):
                if len(where.data.shape) == 1:
                    where = self._backend.index_nplike.nonzero(data_as_index)[0]
                    carry = Index64(where, nplike=self._backend.index_nplike)
                    allow_lazy = "copied"  # True, but also can be modified in-place
                else:
                    wheres = self._backend.index_nplike.nonzero(data_as_index)
                    return self._getitem(wheres, named_axis)
            else:
                raise TypeError(
                    "array slice must be an array of integers or booleans, not\n\n    {}".format(
                        repr(where.data).replace("\n", "\n    ")
                    )
                )

            out = ak._slicing.getitem_next_array_wrap(
                self._carry(carry, allow_lazy), where.shape
            )
            if out.length is not unknown_length and out.length == 0:
                return out._getitem_nothing()
            else:
                return out._getitem_at(0)

        elif isinstance(where, ak.contents.RegularArray):
            maybe_numpy = where.maybe_to_NumpyArray()
            if maybe_numpy is None:
                return self._getitem((where,), named_axis)
            else:
                return self._getitem(maybe_numpy, named_axis)

        # Awkward Array of strings
        elif (
            isinstance(where, Content)
            and where._parameters is not None
            and (where._parameters.get("__array__") in ("string", "bytestring"))
        ):
            return self._getitem_fields(ak.operations.to_list(where))

        elif isinstance(where, ak.contents.EmptyArray):
            return where.to_NumpyArray(np.int64)

        elif isinstance(where, Content):
            return self._getitem((where,), named_axis)

        elif is_sized_iterable(where):
            # Do we have an array
            nplike = nplike_of_obj(where, default=None)
            # We can end up with non-array objects associated with an nplike
            if nplike is not None and nplike.is_own_array(where):
                layout = ak.operations.ak_to_layout._impl(
                    where,
                    allow_record=False,
                    allow_unknown=False,
                    none_policy="error",
                    regulararray=False,
                    use_from_iter=False,
                    primitive_policy="error",
                    string_policy="as-characters",
                )
                return self._getitem(layout, named_axis)

            elif len(where) == 0:
                return self._carry(
                    Index64.empty(0, self._backend.index_nplike),
                    allow_lazy=True,
                )
            # Normally we would be worried about np.array et al. being treated
            # as sized iterables instead of arrays. However, the first two cases
            # here will only iterate over the entire array if it contains strings,
            # at which point we need to visit each item anyway
            elif all(isinstance(x, str) for x in where):
                return self._getitem_fields(list(where))

            else:
                layout = ak.to_backend(
                    ak.operations.ak_to_layout._impl(
                        where,
                        allow_record=False,
                        allow_unknown=False,
                        none_policy="error",
                        regulararray=False,
                        use_from_iter=True,
                        primitive_policy="error",
                        string_policy="as-characters",
                    ),
                    self._backend,
                )
                return self._getitem(layout, named_axis)

        else:
            raise TypeError(
                "only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`), "
                "integer/boolean arrays (possibly with variable-length nested "
                "lists or missing values), field name (str) or names (non-tuple "
                "iterable of str) are valid indices for slicing, not\n\n    "
                + repr(where).replace("\n", "\n    ")
            )

    def _is_getitem_at_placeholder(self) -> bool:
        raise NotImplementedError

    def _is_getitem_at_virtual(self) -> bool:
        return NotADirectoryError

    def _getitem_at(self, where: IndexType):
        raise NotImplementedError

    def _getitem_range(self, start: IndexType, stop: IndexType) -> Content:
        raise NotImplementedError

    def _getitem_field(
        self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    ) -> Content:
        raise NotImplementedError

    def _getitem_fields(
        self, where: list[str], only_fields: tuple[str, ...] = ()
    ) -> Content:
        raise NotImplementedError

    def _getitem_next(
        self,
        head: SliceItem | tuple,
        tail: tuple[SliceItem, ...],
        advanced: Index | None,
    ) -> Content:
        raise NotImplementedError

    def _getitem_next_jagged(
        self,
        slicestarts: Index,
        slicestops: Index,
        slicecontent: Content,
        tail: tuple[SliceItem, ...],
    ) -> Content:
        raise NotImplementedError

    def _carry(self, carry: Index, allow_lazy: bool) -> Content:
        raise NotImplementedError

    def _local_index_axis0(self) -> NumpyArray:
        localindex = Index64.empty(self.length, self._backend.index_nplike)
        self._backend.maybe_kernel_error(
            self._backend["awkward_localindex", np.int64](
                localindex.data,
                localindex.length,
            )
        )
        return ak.contents.NumpyArray(
            localindex.data, parameters=None, backend=self._backend
        )

    def _mergeable_next(self, other: Content, mergebool: bool) -> bool:
        raise NotImplementedError

    def _mergemany(self, others: Sequence[Content]) -> Content:
        raise NotImplementedError

    def _merging_strategy(
        self, others: list[Content]
    ) -> tuple[list[Content], list[Content]]:
        if len(others) == 0:
            raise ValueError(
                "to merge this array with 'others', at least one other must be provided"
            )

        head = [self]
        tail = []

        it_others = iter(others)

        for other in it_others:
            if other.is_indexed or other.is_option or other.is_union:
                tail.append(other)
                tail.extend(it_others)
                break
            else:
                head.append(other)

        assert not any(x.backend.nplike.known_data for x in head + tail) or all(
            x.backend.nplike.known_data for x in head + tail
        )

        return head, tail

    def _local_index(self, axis: int, depth: int):
        raise NotImplementedError

    def _reduce_next(
        self,
        reducer: ak._reducers.Reducer,
        negaxis: int,
        starts: Index,
        shifts: Index | None,
        parents: Index,
        outlength: int,
        mask: bool,
        keepdims: bool,
        behavior: dict | None,
    ):
        raise NotImplementedError

    def _argsort_next(
        self,
        negaxis: int,
        starts: Index,
        shifts: Index | None,
        parents: Index,
        outlength: int,
        ascending: bool,
        stable: bool,
    ):
        raise NotImplementedError

    def _sort_next(
        self,
        negaxis: int,
        starts: Index,
        parents: Index,
        outlength: int,
        ascending: bool,
        stable: bool,
    ):
        raise NotImplementedError

    def _combinations_axis0(
        self,
        n: int,
        replacement: bool,
        recordlookup: list[str] | None,
        parameters: dict | None,
    ):
        size = self.length
        if replacement:
            size = size + (n - 1)
        thisn = n
        if thisn is None or size is None:
            combinationslen = size  # not actually size, just an unknown value
        else:
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

        tocarryraw = self._backend.index_nplike.empty(n, dtype=np.intp)
        tocarry = []
        for i in range(n):
            ptr = Index64.empty(
                combinationslen,
                nplike=self._backend.index_nplike,
                dtype=np.int64,
            )
            tocarry.append(ptr)
            if self._backend.nplike.known_data:
                tocarryraw[i] = ptr.ptr

        toindex = Index64.empty(n, self._backend.index_nplike, dtype=np.int64)
        fromindex = Index64.empty(n, self._backend.index_nplike, dtype=np.int64)

        assert (
            toindex.nplike is self._backend.index_nplike
            and fromindex.nplike is self._backend.index_nplike
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
                self.length,
                1,
            )
        )
        contents = []
        length = ak._util.UNSET
        for ptr in tocarry:
            contents.append(
                ak.contents.IndexedArray.simplified(ptr, self, parameters=None)
            )
            length = contents[-1].length
        assert not (length is ak._util.UNSET and self._backend.nplike.known_data)
        return ak.contents.RecordArray(
            contents, recordlookup, length, parameters=parameters, backend=self._backend
        )

    def _combinations(
        self,
        n: int,
        replacement: bool,
        recordlookup: list[str] | None,
        parameters: dict[str, Any] | None,
        axis: int,
        depth: int,
    ):
        raise NotImplementedError

    def _validity_error(self, path: str) -> str:
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        return self._nbytes_part()

    def purelist_parameter(self, key: str):
        """
        Return the value of the outermost parameter matching `key` in a sequence
        of nested lists, stopping at the first record or tuple layer.

        If a layer has #ak.types.UnionType, the value is only returned if all
        possibilities have the same value.
        """
        return self.form_cls.purelist_parameter(self, key)

    def purelist_parameters(self, *keys: str):
        """
        Return the value of the outermost parameter matching one of `keys` in a sequence
        of nested lists, stopping at the first record or tuple layer.

        If a layer has #ak.types.UnionType, the value is only returned if all
        possibilities have the same value.
        """
        return self.form_cls.purelist_parameters(self, *keys)

    def _is_unique(
        self,
        negaxis: AxisMaybeNone,
        starts: Index,
        parents: Index,
        outlength: int,
    ) -> bool:
        raise NotImplementedError

    def _unique(
        self,
        negaxis: AxisMaybeNone,
        starts: Index,
        parents: Index,
        outlength: int,
    ):
        raise NotImplementedError

    def _pad_none_axis0(self, target: int, clip: bool) -> Content:
        if not clip and (self.length is unknown_length or (target < self.length)):
            index = Index64(
                self._backend.index_nplike.arange(self.length, dtype=np.int64),
                nplike=self._backend.index_nplike,
            )

        else:
            index = Index64.empty(target, self._backend.index_nplike)

            assert index.nplike is self._backend.index_nplike
            self._backend.maybe_kernel_error(
                self._backend[
                    "awkward_index_rpad_and_clip_axis0",
                    index.dtype.type,
                ](index.data, target, self.length)
            )

        return ak.contents.IndexedOptionArray.simplified(index, self)

    def _pad_none(self, target: int, axis: int, depth: int, clip: bool) -> Content:
        raise NotImplementedError

    def to_arrow(
        self,
        list_to32: bool = False,
        string_to32: bool = False,
        bytestring_to32: bool = False,
        emptyarray_to=None,
        categorical_as_dictionary: bool = False,
        extensionarray: bool = True,
        count_nulls: bool = True,
        record_is_scalar: bool = False,
    ):
        import awkward._connect.pyarrow

        pyarrow = awkward._connect.pyarrow.import_pyarrow("to_arrow")
        return self._to_arrow(
            pyarrow,
            None,
            None,
            self.length,
            {
                "list_to32": list_to32,
                "string_to32": string_to32,
                "bytestring_to32": bytestring_to32,
                "emptyarray_to": emptyarray_to,
                "categorical_as_dictionary": categorical_as_dictionary,
                "extensionarray": extensionarray,
                "count_nulls": count_nulls,
                "record_is_scalar": record_is_scalar,
            },
        )

    def _to_arrow(
        self,
        pyarrow: Any,
        mask_node: Content | None,
        validbytes: Content | None,
        length: int,
        options: ToArrowOptions,
    ):
        raise NotImplementedError

    def _to_cudf(self, cudf: Any, mask: Content | None, length: int):
        # prototype abstract signature
        raise NotImplementedError

    def to_backend_array(
        self, allow_missing: bool = True, *, backend: Backend | str | None = None
    ):
        if backend is None:
            backend = self._backend
        else:
            backend = regularize_backend(backend)
        return self._to_backend_array(allow_missing, backend)

    def _to_backend_array(self, allow_missing: bool, backend: Backend):
        raise NotImplementedError

    def drop_none(self):
        return self._drop_none()

    def _drop_none(self) -> Content:
        raise NotImplementedError

    def _remove_structure(
        self, backend: Backend, options: RemoveStructureOptions
    ) -> list[Content]:
        raise NotImplementedError

    def _recursively_apply(
        self,
        action: ImplementsApplyAction,
        depth: int,
        depth_context: Mapping[str, Any] | None,
        lateral_context: Mapping[str, Any] | None,
        options: ApplyActionOptions,
    ) -> Content | None:
        raise NotImplementedError

    def to_json(
        self,
        nan_string: str | None = None,
        posinf_string: str | None = None,
        neginf_string: str | None = None,
        complex_record_fields: tuple[str, str] | None = None,
        convert_bytes: bool | None = None,
        behavior: dict | None = None,
    ) -> list:
        if complex_record_fields is None:
            complex_real_string = None
            complex_imag_string = None
        elif (
            isinstance(complex_record_fields, Sized)
            and isinstance(complex_record_fields, Iterable)
            and len(complex_record_fields) == 2
            and isinstance(complex_record_fields[0], str)
            and isinstance(complex_record_fields[1], str)
        ):
            complex_real_string, complex_imag_string = complex_record_fields
        else:
            complex_real_string, complex_imag_string = None, None

        return self.to_packed()._to_list(
            behavior,
            {
                "nan_string": nan_string,
                "posinf_string": posinf_string,
                "neginf_string": neginf_string,
                "complex_real_string": complex_real_string,
                "complex_imag_string": complex_imag_string,
                "convert_bytes": convert_bytes,
            },
        )

    def to_packed(self, recursive: bool = True) -> Content:
        raise NotImplementedError

    def to_list(self, behavior: dict | None = None) -> list:
        return self.to_packed()._to_list(behavior, None)

    def _to_list(
        self, behavior: dict | None, json_conversions: dict[str, Any] | None
    ) -> list:
        raise NotImplementedError

    def _to_list_custom(
        self, behavior: dict | None, json_conversions: dict[str, Any] | None
    ):
        if self.is_record:
            getitem = get_record_class(self, behavior).__getitem__
            overloaded = getitem is not ak.highlevel.Record.__getitem__ and not getattr(
                getitem, "ignore_in_to_list", False
            )
        else:
            getitem = get_array_class(self, behavior).__getitem__
            overloaded = getitem is not ak.highlevel.Array.__getitem__ and not getattr(
                getitem, "ignore_in_to_list", False
            )

        if overloaded:
            array = wrap_layout(self, behavior=behavior)
            out = [None] * self.length
            for i in range(self.length):
                out[i] = array[i]
                if isinstance(out[i], (ak.highlevel.Array, ak.highlevel.Record)):
                    out[i] = out[i]._layout._to_list(behavior, json_conversions)
                elif hasattr(out[i], "tolist"):
                    out[i] = out[i].tolist()

            # These json_conversions are applied in NumpyArray (for numbers)
            # and ListArray/ListOffsetArray/RegularArray (for bytestrings),
            # but they're also applied here because __getitem__ might return
            # something convertible (the overloaded __getitem__ might be
            # trivial, as it is in Vector).
            if json_conversions is not None:
                convert_bytes = json_conversions["convert_bytes"]
                if convert_bytes is not None:
                    for i, x in enumerate(out):
                        if isinstance(x, bytes):
                            out[i] = convert_bytes(x)

                outimag = None
                complex_real_string = json_conversions["complex_real_string"]
                complex_imag_string = json_conversions["complex_imag_string"]
                if complex_real_string is not None:
                    if any(
                        not isinstance(x, Real) and isinstance(x, Complex) for x in out
                    ):
                        outimag = [None] * len(out)
                        for i, x in enumerate(out):
                            if isinstance(x, Complex):
                                out[i] = x.real
                                outimag[i] = x.imag
                            else:
                                out[i] = x
                                outimag[i] = None

                filters = []

                nan_string = json_conversions["nan_string"]
                if nan_string is not None:
                    isnan = math.isnan
                    filters.append(lambda x: nan_string if isnan(x) else x)

                posinf_string = json_conversions["posinf_string"]
                if posinf_string is not None:
                    inf = float("inf")
                    filters.append(lambda x: posinf_string if x == inf else x)

                neginf_string = json_conversions["neginf_string"]
                if neginf_string is not None:
                    minf = float("-inf")
                    filters.append(lambda x: neginf_string if x == minf else x)

                if len(filters) == 1:
                    f0 = filters[0]
                    for i, x in enumerate(out):
                        out[i] = f0(x)
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f0(x)
                elif len(filters) == 2:
                    f0 = filters[0]
                    f1 = filters[1]
                    for i, x in enumerate(out):
                        out[i] = f1(f0(x))
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f1(f0(x))
                elif len(filters) == 3:
                    f0 = filters[0]
                    f1 = filters[1]
                    f2 = filters[2]
                    for i, x in enumerate(out):
                        out[i] = f2(f1(f0(x)))
                    if outimag is not None:
                        for i, x in enumerate(outimag):
                            outimag[i] = f2(f1(f0(x)))

                if outimag is not None:
                    for i, (real, imag) in enumerate(zip(out, outimag)):
                        out[i] = {complex_real_string: real, complex_imag_string: imag}

            return out

    def _offsets_and_flattened(self, axis: int, depth: int) -> tuple[Index, Content]:
        raise NotImplementedError

    def to_backend(self, backend: Backend | str | None = None) -> Self:
        if backend is None:
            backend = self._backend
        else:
            backend = regularize_backend(backend)
        if backend is self._backend:
            return self
        else:
            return self._to_backend(backend)

    def _to_backend(self, backend: Backend) -> Self:
        raise NotImplementedError

    def with_parameter(self, key: str, value: Any) -> Self:
        out = copy.copy(self)

        if self._parameters is None:
            parameters = {}
        else:
            parameters = copy.copy(self._parameters)
        parameters[key] = value
        out._parameters = parameters

        return out

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def is_equal_to(
        self,
        other: Content,
        index_dtype: bool = True,
        numpyarray: bool = True,
        *,
        all_parameters: bool = False,
    ) -> bool:
        return self._is_equal_to(other, index_dtype, numpyarray, all_parameters)

    def _is_equal_to_generic(self, other: Content, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            self.__class__ is other.__class__
            and (
                self.length is unknown_length
                or other.length is unknown_length
                or self.length == other.length
            )
            and compare_parameters(self._parameters, other._parameters)
        )

    def _is_equal_to(
        self, other: Self, index_dtype: bool, numpyarray: bool, all_parameters: bool
    ) -> bool:
        raise NotImplementedError

    def _repr(self, indent: str, pre: str, post: str) -> str:
        raise NotImplementedError

    def _numbers_to_type(self, name: str, including_unknown: bool) -> Content:
        raise NotImplementedError

    def _fill_none(self, value: Content) -> Content:
        raise NotImplementedError

    def copy(self, *, parameters: JSONMapping | None = UNSET) -> Self:
        raise NotImplementedError

    @classmethod
    def _arrow_needs_option_type(cls):
        return cls.is_option  # is_option is a class property of Meta


@register_backend_lookup_factory
def find_content_backend(obj: type):
    if issubclass(obj, Content):

        def finder(obj: Content):
            return obj.backend

        return finder
