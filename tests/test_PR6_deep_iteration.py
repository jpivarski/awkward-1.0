# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

# py27 = 2 if sys.version_info[0] < 3 else 1

def test_iterator():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray(offsets, content)
    assert list(content) == [1.1, 2.2, 3.3]
    assert [numpy.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]

def test_refcount():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray(offsets, content)

    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter1 = iter(content)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x1 = next(iter1)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter2 = iter(array)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x2 = next(iter2)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter1
    del x1
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter2
    del x2
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

def test_numba_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(12))

    @numba.njit
    def f1(q):
        out = 0.0
        for x in q:
            out += x
        return out

    print(f1(array))
    raise Exception
