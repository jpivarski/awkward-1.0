# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_example():
    x = ak._v2.operations.convert.from_iter([[1, 2, 3, None], [], [4, 5]])
    y = ak._v2.operations.convert.from_iter([100, 200, 300])

    ttx = ak._v2.highlevel.Array(x.layout.typetracer)
    tty = ak._v2.highlevel.Array(y.layout.typetracer)

    assert (x + y).type == (ttx + tty).type
    assert (x + np.sin(y)).type == (ttx + np.sin(tty)).type
