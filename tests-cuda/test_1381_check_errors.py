# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cuda

try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test():
    v2_array = ak.Array([1, 2, 3, 4, 5]).layout

    stream = cp.cuda.Stream(non_blocking=True)

    v2_array_cuda = ak.to_backend(v2_array, "cuda")
    with cp.cuda.Stream() as stream:
        v2_array_cuda[10,]
        v2_array_cuda[11,]
        v2_array_cuda[12,]

    with pytest.raises(ValueError) as err:
        awkward._connect.cuda.synchronize_cuda(stream)

    assert isinstance(err.value, ValueError)

    assert """<Array [1, 2, 3, 4, 5] type='5 * int64'>

with

    (10)

Error details: index out of range in compiled CUDA code (awkward_RegularArray_getitem_next_at)""" in str(
        err.value
    )
