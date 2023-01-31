# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._connect.numpy import unsupported
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


_ZEROS = object()


def zeros_like(array, *, dtype=None, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        dtype (None or NumPy dtype): Overrides the data type of the result.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This is the equivalent of NumPy's `np.zeros_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    with ak._errors.OperationErrorContext(
        "ak.zeros_like",
        {"array": array, "dtype": dtype, "highlevel": highlevel, "behavior": behavior},
    ):
        return _impl(array, highlevel, behavior, dtype)


def _impl(array, highlevel, behavior, dtype):
    if dtype is not None:
        return ak.operations.ak_full_like._impl(array, 0, highlevel, behavior, dtype)
    return ak.operations.ak_full_like._impl(array, _ZEROS, highlevel, behavior, dtype)


@ak._connect.numpy.implements("zeros_like")
def _nep_18_impl(
    a, dtype=None, order=unsupported, subok=unsupported, shape=unsupported
):
    return zeros_like(a, dtype=dtype)
