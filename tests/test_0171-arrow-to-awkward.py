# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

pyarrow = pytest.importorskip("pyarrow")


def test_toarrow():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bitmask = awkward1.layout.IndexU8(numpy.array([40, 34], dtype=numpy.uint8))
    array = awkward1.layout.BitMaskedArray(bitmask, content, False, 9, False)
    assert awkward1.to_arrow(array).to_pylist() == awkward1.to_list(array)

    bytemask = awkward1.layout.Index8(
        numpy.array([False, True, False], dtype=numpy.bool))
    array = awkward1.layout.ByteMaskedArray(bytemask, content, True)
    assert awkward1.to_arrow(array).to_pylist() == awkward1.to_list(array)

    array = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    assert isinstance(awkward1.to_arrow(array),
                      (pyarrow.lib.Tensor, pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == [
        0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    array = awkward1.layout.NumpyArray(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))
    assert isinstance(awkward1.to_arrow(array),
                      (pyarrow.lib.Tensor, pyarrow.lib.Array))
    assert awkward1.to_arrow(array) == pyarrow.Tensor.from_numpy(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))

    array = awkward1.layout.EmptyArray()
    assert isinstance(awkward1.to_arrow(array), (pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == []

    array = awkward1.layout.Index8(numpy.array(
        [1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    assert isinstance(awkward1.to_arrow(array), (pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == [1, 1, 0, 0, 1, 0, 1, 1]

    array = awkward1.layout.IndexU32(numpy.array(
        [0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.uint32))
    assert isinstance(awkward1.to_arrow(array), (pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == [0, 1, 0, 1, 2, 2, 4, 3]

    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray64(offsets, content)
    assert isinstance(awkward1.to_arrow(array), (pyarrow.LargeListArray))
    assert awkward1.to_arrow(array).to_pylist() == [[1.1, 2.2, 3.3], [], [
        4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.IndexU32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArrayU32(offsets, content)
    assert isinstance(awkward1.to_arrow(array), (pyarrow.LargeListArray))
    assert awkward1.to_arrow(array).to_pylist() == [[1.1, 2.2, 3.3], [], [
        4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    # Testing parameters
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert awkward1.to_arrow(array).to_pylist() == [['one', 'two', 'three'], [], [
        'four', 'five'], ['six'], ['seven', 'eight', 'nine']]

    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    assert isinstance(awkward1.to_arrow(listarray), (pyarrow.LargeListArray))
    assert awkward1.to_arrow(listarray).to_pylist() == [[[[0.0, 1.1, 2.2], []], [
        [3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]

    assert isinstance(awkward1.to_arrow(regulararray),
                      (pyarrow.LargeListArray))
    assert awkward1.to_arrow(regulararray).to_pylist() == [[[0.0, 1.1, 2.2], []], [
        [3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))

    recordarray = awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"])

    assert isinstance(awkward1.to_arrow(recordarray), (pyarrow.StructArray))
    assert awkward1.to_arrow(recordarray).to_pylist() == [{'one': 1, 'two': [0.0, 1.1, 2.2], '2': 1.1, 'wonky': 1}, {'one': 2, 'two': [], '2': 2.2, 'wonky': 2}, {
        'one': 3, 'two': [3.3, 4.4], '2': 3.3, 'wonky': 3}, {'one': 4, 'two': [5.5], '2': 4.4, 'wonky': 4}, {'one': 5, 'two': [6.6, 7.7, 8.8, 9.9], '2': 5.5, 'wonky': 5}]

    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    unionarray = awkward1.layout.UnionArray8_32(
        tags, index, [content0, content1])

    assert isinstance(awkward1.to_arrow(unionarray), (pyarrow.UnionArray))
    assert awkward1.to_arrow(unionarray).to_pylist() == [
        1, 2, [1.1, 2.2, 3.3], [], 3, [4.4, 5.5], 5, 4]

    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(
        numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert isinstance(awkward1.to_arrow(indexedarray),
                      (pyarrow.DictionaryArray))
    assert awkward1.to_arrow(indexedarray).to_pylist() == [
        0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array(
        [True, True, False, False, False], dtype=numpy.int8)), listoffsetarray, True)

    assert awkward1.to_arrow(bytemaskedarray).to_pylist() == [
        [0.0, 1.1, 2.2], [], None, None, None]

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), listarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), recordarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

    # bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([True, False, False], dtype=numpy.int8)), indexedarray, True)
    # assert awkward1.to_arrow(bytemaskedarray).to_pylist() == awkward1.to_list(bytemaskedarray)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False, False], dtype=numpy.int8)), unionarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

    ioa = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]), awkward1.layout.NumpyArray(numpy.array([5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
                                                                                                                                                                 6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
                                                                                                                                                                 5.6, 3.8])))
    assert awkward1.to_arrow(ioa).to_pylist() == awkward1.to_list(ioa)
