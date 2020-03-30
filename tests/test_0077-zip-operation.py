# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    content1 = awkward1.Array([1, 2, 3, 4, 5])
    content2 = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert awkward1.tolist(awkward1.zip({"x": content1, "y": content2})) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]
    assert awkward1.tolist(awkward1.zip([content1, content2])) == [(1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4), (5, 5.5)]

    content3 = awkward1.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    content4 = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    assert awkward1.tolist(awkward1.zip({"x": content3, "y": content4})) == [[{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}], [{"x": 5, "y": 5.5}], [{"x": 6, "y": 6.6}, {"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}]]
    assert awkward1.tolist(awkward1.zip({"x": content3, "y": content4}, depthlimit=1)) == [{"x": [0, 1, 2], "y": [0.0, 1.1, 2.2]}, {"x": [], "y": []}, {"x": [3, 4], "y": [3.3, 4.4]}, {"x": [5], "y": [5.5]}, {"x": [6, 7, 8, 9], "y": [6.6, 7.7, 8.8, 9.9]}]

    assert awkward1.tolist(awkward1.zip({"x": content1, "y": content4})) == [[{"x": 1, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 1, "y": 2.2}], [], [{"x": 3, "y": 3.3}, {"x": 3, "y": 4.4}], [{"x": 4, "y": 5.5}], [{"x": 5, "y": 6.6}, {"x": 5, "y": 7.7}, {"x": 5, "y": 8.8}, {"x": 5, "y": 9.9}]]

    assert awkward1.tolist(awkward1.zip({"x": content1, "y": content2, "z": content4})) == [[{"x": 1, "y": 1.1, "z": 0.0}, {"x": 1, "y": 1.1, "z": 1.1}, {"x": 1, "y": 1.1, "z": 2.2}], [], [{"x": 3, "y": 3.3, "z": 3.3}, {"x": 3, "y": 3.3, "z": 4.4}], [{"x": 4, "y": 4.4, "z": 5.5}], [{"x": 5, "y": 5.5, "z": 6.6}, {"x": 5, "y": 5.5, "z": 7.7}, {"x": 5, "y": 5.5, "z": 8.8}, {"x": 5, "y": 5.5, "z": 9.9}]]

    assert awkward1.tolist(awkward1.zip([content1, content2, content3])) == [[(1, 1.1, 0), (1, 1.1, 1), (1, 1.1, 2)], [], [(3, 3.3, 3), (3, 3.3, 4)], [(4, 4.4, 5)], [(5, 5.5, 6), (5, 5.5, 7), (5, 5.5, 8), (5, 5.5, 9)]]
