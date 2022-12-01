# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from typing import Any

import numpy

from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward.typing import Final


class Numpy(ArrayModuleNumpyLike):
    """
    A concrete class importing `NumpyModuleLike` for `numpy`
    """

    is_eager: Final[bool] = True

    array_module: Final[Any] = numpy
