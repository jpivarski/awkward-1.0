# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
from collections.abc import Iterable, Sized
from functools import lru_cache
from types import ModuleType

from packaging.version import parse as parse_version

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import parameters_union

np = NumpyMetadata.instance()
numpy = Numpy.instance()

try:
    import pyarrow

    error_message = None

except ModuleNotFoundError:
    pyarrow = None
    error_message = """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
"""

else:
    if parse_version(pyarrow.__version__) < parse_version("7.0.0"):
        pyarrow = None
        error_message = "pyarrow 7.0.0 or later required for {0}"

if error_message is None:
    from .extn_types import AwkwardArrowArray, AwkwardArrowType
    from .table_conv import (
        convert_awkward_arrow_table_to_native,
        convert_native_arrow_table_to_awkward,
    )
    from .conversions import (
        and_validbytes,
        to_validbits,
        to_length,
        to_null_count,
        to_awkwardarrow_storage_types,
        popbuffers,
        handle_arrow,
        convert_to_array,
        to_awkwardarrow_type,
        direct_Content_subclass,
        direct_Content_subclass_name,
        remove_optiontype,
        form_handle_arrow,
    )
else:
    def nothing_without_pyarrow(*args, **kwargs):
        raise NotImplementedError("This function requires pyarrow, which is not installed.")
    convert_awkward_arrow_table_to_native = nothing_without_pyarrow
    convert_native_arrow_table_to_awkward = nothing_without_pyarrow
    and_validbytes = nothing_without_pyarrow
    to_validbits = nothing_without_pyarrow
    to_length = nothing_without_pyarrow
    to_null_count = nothing_without_pyarrow
    to_awkwardarrow_storage_types = nothing_without_pyarrow
    popbuffers = nothing_without_pyarrow
    handle_arrow = nothing_without_pyarrow
    convert_to_array = nothing_without_pyarrow
    to_awkwardarrow_type = nothing_without_pyarrow
    direct_Content_subclass = nothing_without_pyarrow
    direct_Content_subclass_name = nothing_without_pyarrow
    remove_optiontype = nothing_without_pyarrow
    form_handle_arrow = nothing_without_pyarrow

def import_pyarrow(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))
    return pyarrow


def import_pyarrow_parquet(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.parquet as out

    return out


def import_pyarrow_compute(name: str) -> ModuleType:
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.compute as out

    return out
