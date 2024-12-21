# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import enum
import functools
import itertools
from collections.abc import Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
    _add_named_axis,
    _unify_named_axis,
)
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import (
    parameters_are_empty,
    parameters_are_equal,
    parameters_intersect,
)
from awkward._typing import Any, JSONMapping, List, TypedDict
from awkward._util import UNSET, Sentinel
from awkward.contents.bitmaskedarray import BitMaskedArray
from awkward.contents.bytemaskedarray import ByteMaskedArray
from awkward.contents.content import Content
from awkward.contents.emptyarray import EmptyArray
from awkward.contents.indexedarray import IndexedArray
from awkward.contents.indexedoptionarray import IndexedOptionArray
from awkward.contents.listarray import ListArray
from awkward.contents.listoffsetarray import ListOffsetArray
from awkward.contents.numpyarray import NumpyArray
from awkward.contents.recordarray import RecordArray
from awkward.contents.regulararray import RegularArray
from awkward.contents.unionarray import UnionArray
from awkward.contents.unmaskedarray import UnmaskedArray
from awkward.forms import ByteMaskedForm
from awkward.index import (  # IndexU8,  ; Index32,  ; IndexU32,  ; noqa: F401
    Index8,
    Index64,
)
from awkward.record import Record

np = NumpyMetadata.instance()
numpy = Numpy.instance()

optiontypes = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)
listtypes = (ListOffsetArray, ListArray, RegularArray)


class BroadcastOptions(TypedDict):
    allow_records: bool
    left_broadcast: bool
    right_broadcast: bool
    numpy_to_regular: bool
    regular_to_jagged: bool
    function_name: str | None
    broadcast_parameters_rule: BroadcastParameterRule


def length_of_broadcast(inputs: Sequence) -> int | type[unknown_length]:
    max_length: int | None = None
    has_seen_unknown_length: bool = False
    for x in inputs:
        if not isinstance(x, Content):
            continue
        if x.length is unknown_length:
            has_seen_unknown_length = True
        elif max_length is None:
            max_length = x.length
        else:
            max_length = max(max_length, x.length)

    if has_seen_unknown_length:
        if max_length is None:
            return unknown_length
        else:
            return max_length
    elif max_length is None:
        return 1
    else:
        return max_length


def broadcast_pack(inputs: Sequence, isscalar: list[bool]) -> list:
    maxlen = length_of_broadcast(inputs)
    nextinputs = []

    for x in inputs:
        if isinstance(x, Record):
            index = x.backend.index_nplike.full(maxlen, x.at, dtype=np.int64)
            nextinputs.append(RegularArray(x.array[index], maxlen, 1))
            isscalar.append(True)
        elif isinstance(x, Content):
            nextinputs.append(
                RegularArray(
                    x,
                    x.length,
                    1,
                    parameters=None,
                )
            )
            isscalar.append(False)
        else:
            nextinputs.append(x)
            isscalar.append(True)

    return nextinputs


def broadcast_unpack(x, isscalar: list[bool]):
    if all(isscalar):
        if x.length is not unknown_length and x.length == 0:
            return x._getitem_nothing()._getitem_nothing()
        else:
            return x[0][0]
    else:
        if x.length is not unknown_length and x.length == 0:
            return x._getitem_nothing()
        else:
            return x[0]


def in_function(options):
    if options["function_name"] is None:
        return ""
    else:
        return " in " + options["function_name"]


def ensure_common_length(inputs, options: BroadcastOptions) -> ShapeItem:
    it = iter(inputs)
    length: ShapeItem = unknown_length
    for content in it:
        if content.length is not unknown_length:
            length = content.length
            break

    for other_content in it:
        if other_content.length is unknown_length:
            continue
        if other_content.length != length:
            raise ValueError(
                f"cannot broadcast {type(content).__name__} of length {length} with {type(other_content).__name__} of length {other_content.length}{in_function(options)}"
            )
    return length


def offset_source_rank(list_content: ak.contents.Content) -> int:
    if isinstance(list_content, ak.contents.ListOffsetArray):
        return 0
    elif isinstance(list_content, ak.contents.ListArray):
        return 1
    else:
        raise AssertionError


NO_PARAMETERS = Sentinel("NO_PARAMETERS", __name__)


class BroadcastParameterRule(str, enum.Enum):
    """Behaviour for parameter coalescence during broadcasting."""

    INTERSECT = "intersect"
    ONE_TO_ONE = "one_to_one"
    ALL_OR_NOTHING = "all_or_nothing"
    NONE = "none"


def parameters_of(obj: Any, default: Any = NO_PARAMETERS) -> JSONMapping | None:
    """
    Args:
        obj: #ak.contents.Content that holds parameters, or object
        default: value to return if obj is not an #ak.contents.Content

    Return the parameters of an object if it is a #ak.contents.Content;
    otherwise, return a default value.
    """
    if isinstance(obj, ak.contents.Content):
        return obj._parameters
    else:
        return default


def all_or_nothing_parameters_factory(
    parameters: Sequence[JSONMapping | None], n_outputs: int
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    If the parameters of all the given contents are equal, then the first content's
    parameters are repeated, i.e. `[parameters, parameters, ...]`. Otherwise, a list
    of Nones is returned, i.e. `[None, None, ...]`.
    """
    input_parameters = [p for p in parameters if p is not NO_PARAMETERS]

    result = None
    if len(input_parameters) > 0:
        # All parameters must match this first layout's parameters
        first_parameters = input_parameters[0]
        # Ensure all parameters match, or set parameters to None
        for other_parameters in input_parameters[1:]:
            if not parameters_are_equal(first_parameters, other_parameters):
                break
        else:
            result = first_parameters

    # NB: we don't make unique copies here, so let's hope everyone
    # is well-behaved downstream!
    return [result] * n_outputs


def intersection_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "intersection rule":

    The intersection of `content._parameters.items()` for each content is computed.
    If any parameter dictionaries are None, then a list of Nones is returned, i.e.
    `[None, None, ...]`; otherwise, the computed parameter dictionary is repeated,
    i.e. `[parameters, parameters, ...]`.
    """
    input_parameters = [p for p in parameters if p is not NO_PARAMETERS]

    intersected_parameters = None
    parameters_to_intersect = []
    # Build a list of set-like dict.items() views.
    # If we encounter None-parameters, then we stop early
    # as there can be no intersection.
    for params in input_parameters:
        if parameters_are_empty(params):
            break
        else:
            parameters_to_intersect.append(params)
    # Otherwise, build the intersected parameter dict
    else:
        if len(parameters_to_intersect):
            intersected_parameters = functools.reduce(
                parameters_intersect, parameters_to_intersect
            )
        else:
            intersected_parameters = None

    # NB: we don't make unique copies here, so let's hope everyone
    # is well-behaved downstream!
    return [intersected_parameters] * n_outputs


def one_to_one_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using a "one-to-one rule":

    The requested number of outputs is compared against the length of the given
    `inputs`. If the two values match, then a list of parameter objects is returned,
    where each element of the returned list corresponds to the parameters of the
    content at the same position in the `inputs` sequence. If the length of the
    given contents does not match the requested list length, the intersection of the parameters
    is returned instead.
    """
    if n_outputs == len(parameters):
        return [p if p is not NO_PARAMETERS else None for p in parameters]
    else:
        return intersection_parameters_factory(parameters, n_outputs)


def none_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    A list of Nones is returned, with a length corresponding to the requested number of
    outputs, i.e. `[None, None, ...]`.
    """

    return [None] * n_outputs


def is_string_like(obj) -> bool:
    return (
        isinstance(obj, ak.contents.Content)
        and obj.is_list
        and obj.parameter("__array__") in {"string", "bytestring"}
    )


# Mapping from rule enum values to factory implementations
BROADCAST_RULE_TO_FACTORY_IMPL = {
    BroadcastParameterRule.INTERSECT: intersection_parameters_factory,
    BroadcastParameterRule.ALL_OR_NOTHING: all_or_nothing_parameters_factory,
    BroadcastParameterRule.ONE_TO_ONE: one_to_one_parameters_factory,
    BroadcastParameterRule.NONE: none_parameters_factory,
}


def _export_named_axis_from_depth_to_lateral(
    idx: int,
    depth_context: dict[str, Any],
    lateral_context: dict[str, Any],
) -> None:
    # set adjusted named axes to lateral (inplace)
    named_axis, ndim = depth_context[NAMED_AXIS_KEY][idx]
    seen_named_axis, _ = lateral_context[NAMED_AXIS_KEY][idx]
    lateral_context[NAMED_AXIS_KEY][idx] = (
        _unify_named_axis(named_axis, seen_named_axis),
        ndim,
    )


def broadcast_regular_dim_size(contents: Sequence[ak.contents.Content]) -> ShapeItem:
    # Find known size out of our contents
    dim_size: ShapeItem
    it_non_string_regular_contents = iter(
        c for c in contents if c.is_regular and not is_string_like(c)
    )
    for x in it_non_string_regular_contents:
        if x.size is not unknown_length:
            dim_size = x.size
            break
    else:
        # We should only be here if we didn't find any regular arrays with known lengths;
        # there is guaranteed to be at least one non-string list
        dim_size = unknown_length
    # Now we know that we have at least one layout with concrete size, let's check the remainder
    # dim_size=0 should win, though, so we require that dim_size != 0
    if dim_size is not unknown_length and dim_size > 0:
        for x in it_non_string_regular_contents:
            # Any unknown lengths can't be compared
            if x.size is unknown_length:
                continue
            # Any zero-length column triggers zero broadcasting
            if x.size == 0:
                return 0
            else:
                dim_size = max(dim_size, x.size)
    return dim_size


def broadcast_to_offsets_avoiding_carry(
    list_content: ak.contents.Content,
    offsets: ak.index.Index,
) -> ak.contents.Content:
    index_nplike = list_content.backend.index_nplike

    # Without known data, we can't perform these optimisations
    if not index_nplike.known_data:
        return list_content._broadcast_tooffsets64(offsets).content

    elif isinstance(list_content, ListOffsetArray):
        if index_nplike.array_equal(offsets.data, list_content.offsets.data):
            next_length = index_nplike.index_as_shape_item(offsets[-1])
            return list_content.content[:next_length]
        else:
            return list_content._broadcast_tooffsets64(offsets).content
    elif isinstance(list_content, ListArray):
        # Is this list contiguous?
        if index_nplike.array_equal(
            list_content.starts.data[1:], list_content.stops.data[:-1]
        ):
            # Does this list match the offsets?
            if index_nplike.array_equal(
                offsets.data[:-1], list_content.starts.data
            ) and not (
                list_content.stops.data.shape[0] != 0
                and offsets[-1] != list_content.stops[-1]
            ):
                next_length = index_nplike.index_as_shape_item(offsets[-1])
                return list_content.content[:next_length]
            else:
                return list_content._broadcast_tooffsets64(offsets).content
        else:
            return list_content._broadcast_tooffsets64(offsets).content

    elif isinstance(list_content, RegularArray):
        my_offsets = list_content._compact_offsets64(True)
        if index_nplike.array_equal(offsets.data, my_offsets.data):
            return list_content.content[: list_content.size * list_content.length]
        else:
            return list_content._broadcast_tooffsets64(offsets).content

    else:
        raise AssertionError


def apply_step(
    backend: Backend,
    inputs: Sequence,
    action,
    depth: int,
    depth_context,
    lateral_context,
    options: BroadcastOptions,
):
    # This happens when descending anyway, but setting the option does it before action.
    if options["numpy_to_regular"] and any(
        isinstance(x, NumpyArray) and x._data.ndim != 1 for x in inputs
    ):
        inputs = [
            x.to_RegularArray() if isinstance(x, NumpyArray) else x for x in inputs
        ]

    # Rare that any function would want this, but some do.
    if options["regular_to_jagged"] and any(
        isinstance(x, RegularArray) for x in inputs
    ):
        inputs = [
            x.to_ListOffsetArray64(False) if isinstance(x, RegularArray) else x
            for x in inputs
        ]

    contents = [x for x in inputs if isinstance(x, Content)]

    # Handle implicit right-broadcasting (NumPy-like broadcasting).
    if options["right_broadcast"] and any(isinstance(x, listtypes) for x in inputs):
        max_depth = max(x.purelist_depth for x in contents)

        if max_depth > 0 and all(x.purelist_isregular for x in contents):
            nextinputs = []

            named_axes_with_ndims = depth_context[NAMED_AXIS_KEY]
            seen_named_axes = lateral_context[NAMED_AXIS_KEY]
            for i, ((named_axis, ndim), o) in enumerate(
                zip(named_axes_with_ndims, inputs)
            ):
                if isinstance(o, Content):
                    # rightbroadcast
                    for _ in range(o.purelist_depth, max_depth):
                        o = RegularArray(o, 1, o.length)
                        # track new dimensions for named axis
                        # rightbroadcasting adds a new first(!) dimension at depth
                        seen_named_axis, seen_ndim = seen_named_axes[i]
                        named_axis = _add_named_axis(named_axis, depth, ndim)
                        depth_context[NAMED_AXIS_KEY][i] = (
                            _unify_named_axis(named_axis, seen_named_axis),
                            ndim + 1 if ndim is not None else ndim,
                        )
                        if o.is_leaf:
                            _export_named_axis_from_depth_to_lateral(
                                i, depth_context, lateral_context
                            )
                    nextinputs.append(o)
                else:
                    nextinputs.append(o)
            # Did a broadcast take place?
            if any(x is not y for x, y in zip(inputs, nextinputs)):
                return apply_step(
                    backend,
                    nextinputs,
                    action,
                    depth,
                    depth_context,
                    lateral_context,
                    options,
                )

    # Now all lengths must agree
    length = ensure_common_length(contents, options)

    # Load the parameter broadcasting rule implementation
    rule = options["broadcast_parameters_rule"]
    try:
        parameters_factory = BROADCAST_RULE_TO_FACTORY_IMPL[rule]
    except KeyError:
        raise ValueError(
            f"`broadcast_parameters_rule` should be one of {[str(x) for x in BroadcastParameterRule]}, "
            f"but this routine received `{rule}`"
        ) from None

    # This whole function is one big switch statement.
    def broadcast_any_record():
        if not options["allow_records"]:
            raise ValueError(f"cannot broadcast records{in_function(options)}")

        frozen_record_fields: frozenset[str] | None = UNSET
        first_record = next(c for c in contents if c.is_record)
        nextparameters = []

        for x in contents:
            if x.is_record:
                nextparameters.append(x._parameters)

                # Ensure all records are tuples, or all records are records
                if x.is_tuple != first_record.is_tuple:
                    raise TypeError(
                        f"cannot broadcast a tuple against a record{in_function(options)}"
                    )

                # Check fields match for records
                if not x.is_tuple:
                    if frozen_record_fields is UNSET:
                        frozen_record_fields = frozenset(x.fields)
                    elif frozen_record_fields != frozenset(x.fields):
                        raise ValueError(
                            "cannot broadcast records because fields don't "
                            "match{}:\n    {}\n    {}".format(
                                in_function(options),
                                ", ".join(sorted(first_record.fields)),
                                ", ".join(sorted(x.fields)),
                            )
                        )
            else:
                nextparameters.append(NO_PARAMETERS)

        numoutputs = None
        outcontents = []
        for field in first_record.fields:
            outcontents.append(
                apply_step(
                    backend,
                    [
                        x.content(field) if isinstance(x, RecordArray) else x
                        for x in inputs
                    ],
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )
            )
            assert isinstance(outcontents[-1], tuple)
            if numoutputs is None:
                numoutputs = len(outcontents[-1])
            else:
                assert numoutputs == len(outcontents[-1])

        parameters = parameters_factory(nextparameters, numoutputs)

        return tuple(
            RecordArray(
                [x[i] for x in outcontents],
                # Explicitly set fields to None if this is a tuple
                None if first_record.is_tuple else first_record.fields,
                length,
                parameters=p,
            )
            for i, p in enumerate(parameters)
        )

    def broadcast_any_list():
        index_nplike = backend.index_nplike
        # Under the category of "is_list", we have both strings and non-strings
        # The strings should behave like non-lists within these routines.

        named_axes_with_ndims = depth_context[NAMED_AXIS_KEY]
        # Are the non-string list types exclusively regular?
        if all(x.is_regular or (is_string_like(x) or not x.is_list) for x in contents):
            # Compute the expected dim size
            dim_size = broadcast_regular_dim_size(contents)
            dimsize_maybe_broadcastable = dim_size is unknown_length or dim_size > 1
            dimsize_is_zero = dim_size is not unknown_length and dim_size == 0

            # Build a broadcast index for size=1 contents, and identify whether we have strings
            inputs_are_strings = []
            size_one_carry_index = None
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    content_is_string = is_string_like(x)
                    inputs_are_strings.append(content_is_string)
                    if (
                        # Strings don't count as lists in this context
                        not content_is_string
                        # Is this layout known to be size==1?
                        and x.is_regular
                        and x.size is not unknown_length
                        and x.size == 1
                        # Does the computed dim_size support broadcasting
                        and dimsize_maybe_broadcastable
                        and size_one_carry_index is None
                    ):
                        # For any (N, 1) array, we know we'll broadcast to (N, M) where M is maxsize
                        size_one_carry_index = Index64(
                            index_nplike.repeat(
                                index_nplike.arange(
                                    index_nplike.shape_item_as_index(length),
                                    dtype=np.int64,
                                ),
                                index_nplike.shape_item_as_index(dim_size),
                            ),
                            nplike=index_nplike,
                        )
                else:
                    inputs_are_strings.append(False)

            # W.r.t broadcasting against other lists, we have three possibilities
            # a. any (exactly) size-0 content broadcasts all other regular dimensions to 0
            # b. any (exactly) size-1 content broadcasts to the `size_one_carry_index``
            # c. otherwise, the list size should equal the dimension; recurse into the content as-is

            # If we have non-lists, these are just appended as-is. As we're dealing with regular layouts,
            # we don't left-broadcast
            nextinputs = []
            nextparameters = []
            for i, ((named_axis, ndim), x, x_is_string) in enumerate(
                zip(named_axes_with_ndims, inputs, inputs_are_strings)
            ):
                if isinstance(x, RegularArray) and not x_is_string:
                    content_size_maybe_one = (
                        x.size is not unknown_length and x.size == 1
                    )
                    # If dimsize is known to be exactly zero, all contents are zero length
                    if dimsize_is_zero:
                        nextinputs.append(x.content[:0])
                        nextparameters.append(x._parameters)
                    # If we have a known size=1 content, then broadcast it to the dimension size
                    elif dimsize_maybe_broadcastable and content_size_maybe_one:
                        nextinputs.append(
                            x.content[: x.length * x.size]._carry(
                                size_one_carry_index, allow_lazy=False
                            )
                        )
                        nextparameters.append(x._parameters)
                        # track new dimensions for named axis
                        # rightbroadcasting adds a new first(!) dimension as depth
                        depth_context[NAMED_AXIS_KEY][i] = (
                            _add_named_axis(named_axis, depth, ndim),
                            ndim + 1 if ndim is not None else ndim,
                        )
                        if x.is_leaf:
                            _export_named_axis_from_depth_to_lateral(
                                i, depth_context, lateral_context
                            )
                    # Any unknown values or sizes are assumed to be correct as-is
                    elif (
                        dim_size is unknown_length
                        or x.size is unknown_length
                        or x.size == dim_size
                    ):
                        nextinputs.append(x.content[: x.length * x.size])
                        nextparameters.append(x._parameters)
                    else:
                        raise ValueError(
                            "cannot broadcast RegularArray of size "
                            f"{x.size} with RegularArray of size {dim_size}{in_function(options)}"
                        )
                else:
                    nextinputs.append(x)
                    nextparameters.append(NO_PARAMETERS)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(nextparameters, len(outcontent))

            return tuple(
                RegularArray(x, dim_size, length, parameters=p)
                for x, p in zip(outcontent, parameters)
            )
        # General list-handling case: the offsets of each list may be different.
        else:
            offsets_content = None
            input_is_string = []

            # Find the offsets to broadcast to, taking the "best" list
            for x in inputs:
                if isinstance(x, Content):
                    content_is_string = is_string_like(x)
                    input_is_string.append(content_is_string)
                    if (
                        x.is_list
                        and not x.is_regular
                        and not content_is_string
                        and (
                            offsets_content is None
                            or (
                                offset_source_rank(x)
                                < offset_source_rank(offsets_content)
                            )
                        )
                    ):
                        offsets_content = x
                else:
                    input_is_string.append(False)

            # Build the offsets of the lowest-ranking source
            offsets = offsets_content._compact_offsets64(True)

            nextinputs = []
            nextparameters = []
            for i, ((named_axis, ndim), x, x_is_string) in enumerate(
                zip(named_axes_with_ndims, inputs, input_is_string)
            ):
                if isinstance(x, listtypes) and not x_is_string:
                    next_content = broadcast_to_offsets_avoiding_carry(x, offsets)
                    nextinputs.append(next_content)
                    nextparameters.append(x._parameters)
                # Handle implicit left-broadcasting (non-NumPy-like broadcasting).
                elif options["left_broadcast"] and isinstance(x, Content):
                    nextinputs.append(
                        RegularArray(x, 1, x.length)
                        ._broadcast_tooffsets64(offsets)
                        .content
                    )
                    nextparameters.append(NO_PARAMETERS)
                    # track new dimensions for named axis
                    # leftbroadcasting adds a new last dimension at depth + 1
                    depth_context[NAMED_AXIS_KEY][i] = (
                        _add_named_axis(named_axis, depth + 1, ndim),
                        ndim + 1 if ndim is not None else ndim,
                    )
                    if x.is_leaf:
                        _export_named_axis_from_depth_to_lateral(
                            i, depth_context, lateral_context
                        )
                else:
                    nextinputs.append(x)
                    nextparameters.append(NO_PARAMETERS)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(nextparameters, len(outcontent))

            return tuple(
                ListOffsetArray(offsets, x, parameters=p)
                for x, p in zip(outcontent, parameters)
            )

    def broadcast_any_option_all_UnmaskedArray():
        nextinputs = []
        nextparameters = []
        for x in inputs:
            if isinstance(x, UnmaskedArray):
                nextinputs.append(x.content)
                nextparameters.append(x._parameters)
            elif isinstance(x, Content):
                nextinputs.append(x)
                nextparameters.append(x._parameters)
            else:
                nextinputs.append(x)
                nextparameters.append(NO_PARAMETERS)

        outcontent = apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert isinstance(outcontent, tuple)
        parameters = parameters_factory(nextparameters, len(outcontent))

        return tuple(
            UnmaskedArray(x, parameters=p) for x, p in zip(outcontent, parameters)
        )

    def broadcast_any_option():
        mask = None
        for x in contents:
            if x.is_option:
                m = x.mask_as_bool(valid_when=False)
                if mask is None:
                    mask = m
                else:
                    mask = backend.index_nplike.logical_or(mask, m, maybe_out=mask)

        nextmask = Index8(mask.view(np.int8))
        index = backend.index_nplike.full(mask.shape[0], np.int64(-1), dtype=np.int64)
        index[~mask] = backend.index_nplike.arange(
            backend.index_nplike.shape_item_as_index(mask.shape[0])
            - backend.index_nplike.count_nonzero(mask),
            dtype=np.int64,
        )
        index = Index64(index)
        if any(not x.is_option for x in contents):
            nextindex = backend.index_nplike.arange(
                backend.index_nplike.shape_item_as_index(mask.shape[0]),
                dtype=np.int64,
            )
            nextindex[mask] = -1
            nextindex = Index64(nextindex)

        nextinputs = []
        nextparameters = []
        for x in inputs:
            if isinstance(x, optiontypes):
                nextinputs.append(x.project(nextmask))
                nextparameters.append(x._parameters)
            elif isinstance(x, Content):
                nextinputs.append(IndexedOptionArray(nextindex, x).project(nextmask))
                nextparameters.append(x._parameters)
            else:
                nextinputs.append(x)
                nextparameters.append(NO_PARAMETERS)

        outcontent = apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert isinstance(outcontent, tuple)
        parameters = parameters_factory(nextparameters, len(outcontent))

        return tuple(
            IndexedOptionArray.simplified(index, x, parameters=p)
            for x, p in zip(outcontent, parameters)
        )

    def broadcast_any_option_akwhere():
        """
        ak_where is a bit like the ternary operator. Due to asymmetries in the three
        inputs (their roles are distinct), special handling is required for option-types.
        """
        unmasked = []  # Contents of inputs-as-ByteMaskedArrays or non-Content-type
        masks: List[Index8] = []
        # Here we choose the convention that elements are masked when mask==1
        # And byte masks (not bits) so we can pass them as (x,y) to ak_where's action()
        for xyc in inputs:  # from ak_where, inputs are (x, y, condition)
            if not isinstance(xyc, Content):
                unmasked.append(xyc)
                masks.append(
                    NumpyArray(backend.nplike.zeros(inputs[2].length, dtype=np.int8))
                )
            elif not xyc.is_option:
                unmasked.append(xyc)
                masks.append(
                    NumpyArray(backend.nplike.zeros(xyc.length, dtype=np.int8))
                )
            elif xyc.is_indexed:
                # Indexed arrays have no array elements where None, which is a problem for us.
                # We don't care what the element's value is when masked. Just that there *is* a value.
                if xyc.content.is_unknown:
                    # Unknown arrays cannot use to_ByteMaskedArray.
                    # Create a stand-in array of similar shape and any dtype (we use bool here)
                    unused_unmasked = NumpyArray(
                        backend.nplike.zeros(xyc.length, dtype=np.bool_)
                    )
                    unmasked.append(unused_unmasked)
                    all_masked = NumpyArray(
                        backend.nplike.ones(xyc.length, dtype=np.int8)
                    )
                    masks.append(all_masked)
                else:
                    xyc_as_masked = xyc.to_ByteMaskedArray(valid_when=False)
                    unmasked.append(xyc_as_masked.content)
                    masks.append(NumpyArray(xyc_as_masked.mask.data))
            elif not isinstance(xyc.form, ByteMaskedForm) or xyc.form.valid_when:
                # Must make existing mask conform to our convention
                xyc_as_bytemasked = xyc.to_ByteMaskedArray(valid_when=False)
                unmasked.append(xyc_as_bytemasked.content)
                masks.append(NumpyArray(xyc_as_bytemasked.mask.data))
            else:
                unmasked.append(xyc.content)
                masks.append(NumpyArray(xyc.mask.data))

        # (1) Apply ak_where action to unmasked inputs
        outcontent = apply_step(
            backend,
            unmasked,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert isinstance(outcontent, tuple) and len(outcontent) == 1
        xy_unmasked = outcontent[0]

        # (2) Now apply ak_where action to unmasked condition and mask arrays for x and y
        which_mask = (
            masks[0],  # Now x is the x-mask
            masks[1],  # y-mask
            unmasked[2],  # But same condition as previous
        )
        outmasks = apply_step(
            backend,
            which_mask,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert len(outmasks) == 1
        xy_mask = outmasks[0]

        simple_options = BroadcastOptions(
            allow_records=True,
            left_broadcast=True,
            right_broadcast=True,
            numpy_to_regular=True,
            regular_to_jagged=False,
            function_name=None,
            broadcast_parameters_rule=BroadcastParameterRule.INTERSECT,
        )

        # (3) Since condition may be tree-like, use apply_step to OR condition and result masks
        def action_logical_or(inputs, backend, **kwargs):
            # Return None when condition is None or selected element is None
            m1, m2 = inputs
            if all(isinstance(x, NumpyArray) for x in inputs):
                out = NumpyArray(backend.nplike.logical_or(m1.data, m2.data))
                return (out,)

        cond_mask = masks[2]
        mask = apply_step(
            backend,
            (xy_mask, cond_mask),
            action_logical_or,
            0,
            depth_context,
            lateral_context,
            simple_options,
        )[0]

        # (4) Apply mask to unmasked selection results, recursively
        def apply_mask_action(inputs, backend, **kwargs):
            if all(
                x.is_leaf or (x.branch_depth == (False, 1) and is_string_like(x))
                for x in inputs
            ):
                content, mask = inputs
                if hasattr(mask, "content"):
                    mask_as_idx = Index8(mask.content.data)
                else:
                    mask_as_idx = Index8(mask.data)
                out = ByteMaskedArray(
                    mask_as_idx,
                    content,
                    valid_when=False,
                )
                return (out,)

        masked = apply_step(
            backend,
            (xy_unmasked, mask),
            apply_mask_action,
            0,
            depth_context,
            lateral_context,
            simple_options,
        )
        return masked

    def broadcast_any_union():
        nextparameters = []

        for x in inputs:
            if isinstance(x, UnionArray):
                nextparameters.append(x._parameters)
            else:
                nextparameters.append(NO_PARAMETERS)

        union_tags, union_num_contents, length = [], [], unknown_length
        for x in contents:
            if x.is_union:
                tags = x.tags.raw(backend.index_nplike)
                union_tags.append(tags)
                union_num_contents.append(len(x.contents))

                if length is unknown_length:
                    length = tags.shape[0]
                elif tags.shape[0] is unknown_length:
                    continue
                elif length != tags.shape[0]:
                    raise ValueError(
                        f"cannot broadcast UnionArray of length {length} "
                        f"with UnionArray of length {tags.shape[0]}{in_function(options)}"
                    )

        tags = backend.index_nplike.empty(length, dtype=np.int8)
        index = backend.index_nplike.empty(length, dtype=np.int64)

        # Stack all union tags
        combos = backend.index_nplike.stack(union_tags, axis=-1)

        # Build array of indices (c1, c2, c3, ..., cn) of contents in
        # (union 1, union 2, union 3, ..., union n)
        all_combos = list(itertools.product(*[range(x) for x in union_num_contents]))

        numoutputs = None
        outcontents = []

        for tag, j_contents in enumerate(all_combos):
            combo = backend.index_nplike.asarray(j_contents, dtype=np.int64)
            mask = backend.index_nplike.all(combos == combo, axis=-1)
            tags[mask] = tag
            index[mask] = backend.index_nplike.arange(
                backend.index_nplike.count_nonzero(mask), dtype=np.int64
            )
            nextinputs = []
            it_j_contents = iter(j_contents)
            for x in inputs:
                if isinstance(x, UnionArray):
                    nextinputs.append(x[mask].project(next(it_j_contents)))
                elif isinstance(x, Content):
                    nextinputs.append(x[mask])
                else:
                    nextinputs.append(x)
            outcontents.append(
                apply_step(
                    backend,
                    nextinputs,
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )
            )
            assert isinstance(outcontents[-1], tuple)
            if numoutputs is None:
                numoutputs = len(outcontents[-1])
            else:
                assert numoutputs == len(outcontents[-1])

        assert numoutputs is not None

        parameters = parameters_factory(nextparameters, numoutputs)

        return tuple(
            UnionArray.simplified(
                Index8(tags),
                Index64(index),
                [x[i] for x in outcontents],
                parameters=p,
            )
            for i, p in enumerate(parameters)
        )

    def broadcast_any_indexed():
        # The `apply` function may exit at the level of a `RecordArray`. We can avoid projection
        # of the record array in such cases, in favour of a deferred carry. This can be done by
        # "pushing" the `IndexedArray` _into_ the record (i.e., wrapping each `content`).
        nextinputs = [
            x._push_inside_record_or_project() if isinstance(x, IndexedArray) else x
            for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def broadcast_any_nd_numpy():
        nextinputs = [
            x.to_RegularArray() if isinstance(x, NumpyArray) else x for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def broadcast_any_unknown():
        nextinputs = [
            x.to_NumpyArray(np.float64, backend) if isinstance(x, EmptyArray) else x
            for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def continuation():
        # Any EmptyArrays?
        if any(x.is_unknown for x in contents):
            return broadcast_any_unknown()

        # Any NumpyArrays with ndim != 1?
        elif any(x.is_numpy and x.purelist_depth != 1 for x in contents):
            return broadcast_any_nd_numpy()

        # Any IndexedArrays?
        elif any((x.is_indexed and not x.is_option) for x in contents):
            return broadcast_any_indexed()

        # Any UnionArrays?
        elif any(x.is_union for x in contents):
            return broadcast_any_union()

        # Any option-types?
        elif any(x.is_option for x in contents):
            if all(not x.is_option or isinstance(x, UnmaskedArray) for x in contents):
                return broadcast_any_option_all_UnmaskedArray()
            elif options["function_name"] == "ak.where":
                return broadcast_any_option_akwhere()
            else:
                return broadcast_any_option()

        # Any non-string list-types?
        elif any(x.is_list and not is_string_like(x) for x in contents):
            return broadcast_any_list()

        # Any RecordArrays?
        elif any(x.is_record for x in contents):
            return broadcast_any_record()

        else:
            raise ValueError(
                "cannot broadcast: {}{}".format(
                    ", ".join(repr(type(x)) for x in inputs), in_function(options)
                )
            )

    result = action(
        inputs,
        depth=depth,
        depth_context=depth_context,
        lateral_context=lateral_context,
        continuation=continuation,
        backend=backend,
        options=options,
    )

    if isinstance(result, tuple) and all(isinstance(x, Content) for x in result):
        if any(content.backend is not backend for content in result):
            raise ValueError(
                "broadcasting action returned layouts with different backends: ",
                ", ".join([content.backend.name for content in result]),
            )
        return result
    elif result is None:
        return continuation()
    else:
        raise AssertionError(result)


def broadcast_and_apply(
    inputs,
    action,
    *,
    depth_context: dict[str, Any] | None = None,
    lateral_context: dict[str, Any] | None = None,
    allow_records: bool = True,
    left_broadcast: bool = True,
    right_broadcast: bool = True,
    numpy_to_regular: bool = False,
    regular_to_jagged: bool = False,
    function_name: str | None = None,
    broadcast_parameters_rule=BroadcastParameterRule.INTERSECT,
):
    # Expect arrays to already have common backend
    backend = backend_of(*inputs, coerce_to_common=False)
    isscalar = []
    out = apply_step(
        backend,
        broadcast_pack(inputs, isscalar),
        action,
        0,
        depth_context,
        lateral_context,
        {
            "allow_records": allow_records,
            "left_broadcast": left_broadcast,
            "right_broadcast": right_broadcast,
            "numpy_to_regular": numpy_to_regular,
            "regular_to_jagged": regular_to_jagged,
            "function_name": function_name,
            "broadcast_parameters_rule": broadcast_parameters_rule,
        },
    )
    assert isinstance(out, tuple)
    return tuple(broadcast_unpack(x, isscalar) for x in out)
