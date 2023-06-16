# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("min",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._connect.numpy import UNSUPPORTED
from awkward._errors import with_operation_context
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis

np = NumpyMetadata.instance()


@with_operation_context
def min(
    array,
    axis=None,
    *,
    keepdims=False,
    initial=None,
    mask_identity=True,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        initial (None or number): The maximum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. infinity
            for floating-point types, a maximum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns the minimum value in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The identity
    of minimization is `inf` if floating-point or the largest integer value
    if applied to integers. This identity is usually masked: the minimum of
    an empty list is None, unless `mask_identity=False`.
    This operation is the same as NumPy's
    [amin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    See also #ak.nanmin.
    """
    return _impl(
        array,
        axis,
        keepdims,
        initial,
        mask_identity,
        highlevel,
        behavior,
    )


@with_operation_context
def nanmin(
    array,
    axis=None,
    *,
    keepdims=False,
    initial=None,
    mask_identity=True,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        initial (None or number): The maximum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. infinity
            for floating-point types, a maximum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Like #ak.min, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.min(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.min.
    """
    return _impl(
        ak.operations.ak_nan_to_none._impl(array, False, None),
        axis,
        keepdims,
        initial,
        mask_identity,
        highlevel,
        behavior,
    )


def _impl(array, axis, keepdims, initial, mask_identity, highlevel, behavior):
    axis = regularize_axis(axis)
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = behavior_of(array, behavior=behavior)
    reducer = ak._reducers.Min(initial)

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=behavior,
    )
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return wrap_layout(out, behavior, highlevel)
    else:
        return out


@ak._connect.numpy.implements("amin")
def _nep_18_impl_amin(
    a,
    axis=None,
    out=UNSUPPORTED,
    keepdims=False,
    initial=None,
    where=UNSUPPORTED,
):
    return min(a, axis=axis, keepdims=keepdims, initial=initial)


@ak._connect.numpy.implements("nanmin")
def _nep_18_impl_nanmin(
    a,
    axis=None,
    out=UNSUPPORTED,
    keepdims=False,
    initial=None,
    where=UNSUPPORTED,
):
    return nanmin(a, axis=axis, keepdims=keepdims, initial=initial)
