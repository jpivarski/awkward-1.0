# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak.from_numpy(np.empty((5, 0), dtype=np.float64), regulararray=True)
    y = ak.from_numpy(np.empty((8, 0), dtype=np.float64), regulararray=True)
    z = ak.concatenate([x, y], axis=0)
    assert len(z) == 13
