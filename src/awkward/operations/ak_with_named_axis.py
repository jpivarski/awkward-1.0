# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    AxisMapping,
    AxisTuple,
    _axis_tuple_to_mapping,
    _NamedAxisKey,
)
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("with_named_axis",)

np = NumpyMetadata.instance()


@high_level_function()
def with_named_axis(
    array,
    named_axis: AxisTuple | AxisMapping,
    *,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        named_axis: AxisTuple | AxisMapping: Names to give to the array axis; this assigns
            the `"__named_axis__"` attr. If None, any existing name is unset.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new name. This function does not change the
    array in-place. If the new name is None, then an array without a name is
    returned.

    The records or tuples may be nested within multiple levels of nested lists.
    If records are nested within records, only the outermost are affected.

    Setting the `"__record__"` parameter makes it possible to add behaviors
    to the data; see #ak.Array and #ak.behavior for a more complete
    description.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, named_axis, highlevel, behavior, attrs)


def _impl(array, named_axis, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False)

    # Named axis handling
    if not named_axis:  # no-op, e.g. named_axis is None, (), {}
        _named_axis = {}
    elif isinstance(named_axis, tuple):
        ndim = layout.purelist_depth
        if len(named_axis) != ndim:
            raise ValueError(
                f"{named_axis=} must have the same length as the number of dimensions ({ndim})"
            )
        _named_axis = _axis_tuple_to_mapping(named_axis)
    elif isinstance(named_axis, dict):
        _named_axis = named_axis
    else:
        raise TypeError(f"named_axis must be a mapping or a tuple, got {named_axis}")

    if _named_axis:
        ctx = ctx.with_attr(
            key=_NamedAxisKey,
            value=_named_axis,
        )

    return ctx.wrap(
        layout,
        highlevel=highlevel,
        allow_other=True,
    )
