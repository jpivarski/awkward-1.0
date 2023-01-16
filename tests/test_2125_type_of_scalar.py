# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    assert ak.type(np.int16()) == ak.types.ScalarType(ak.types.NumpyType("int16"))
    assert ak.type(np.uint32) == ak.types.ScalarType(ak.types.NumpyType("uint32"))
    assert ak.type(np.dtype("complex128")) == ak.types.ScalarType(
        ak.types.NumpyType("complex128")
    )
    assert ak.type("hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}), 5
    )
    assert ak.type("int16") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}), 5
    )
    assert ak.type(["int16"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType(
                "uint8", parameters={"__array__": "char"}, typestr="char"
            ),
            parameters={"__array__": "string"},
            typestr="string",
        ),
        1,
    )
