# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("is_categorical",)
import awkward as ak
from awkward._dispatch import high_level_function


def _dispatcher(array):
    yield array


@high_level_function(_dispatcher)
def is_categorical(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    If the `array` is categorical (contains #ak.contents.IndexedArray or
    #ak.contents.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns True;
    otherwise, it returns False.

    See also #ak.categories, #ak.to_categorical, #ak.from_categorical.
    """
    return _impl(array)


def _impl(array):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    return layout.purelist_parameter("__array__") == "categorical"
