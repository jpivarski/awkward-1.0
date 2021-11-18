# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward.forth import ForthMachine32


# FIXME: 'textint->' should not consume whitespace; but explicit skip-ws in


def test_textint():
    vm = ForthMachine32(
        "input x output y float64  3 0 do x textint-> stack loop x textint-> y"
    )
    vm.run({"x": b"     12345 -123      3210  42"})
    assert vm.stack == [12345, -123, 3210]
    assert np.asarray(vm["y"]).tolist() == [42.0]


def test_skip_ws():
    vm = ForthMachine32("input x  x skip-ws")
    vm.run({"x": np.array([b"     hello"])})
    assert vm.input_position("x") == 5
