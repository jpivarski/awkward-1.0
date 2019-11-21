# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_nothing():
    recordarray = awkward1.layout.RecordArray()
    recordarray.append(awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5])), "one")
    recordarray.append(awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])), "two")
    recordarray.alias(0, "wonky")
    assert awkward1.tolist(recordarray.content(0)) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(recordarray.content("two")) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(recordarray.content("wonky")) == [1, 2, 3, 4, 5]
