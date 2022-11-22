# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

to_list = ak.operations.to_list


def test_from_iter():
    a = ak.Array([[1], [2, None]])
    assert to_list(ak.drop_none(a)) == [[1], [2]]

    a = ak.Array([[2, None]])
    assert to_list(ak.drop_none(a)) == [[2]]

    a = ak.Array([[[None]]])
    assert to_list(ak.drop_none(a)) == [[[]]]

    a = ak.Array([1, 2, None])
    assert to_list(ak.drop_none(a, axis=0))

    a = ak.Array([1, 2, 3, None, [None]])
    assert to_list(ak.drop_none(a, axis=1)) == [1, 2, 3, [None]]

    a = ak.Array([[[1, None]], [[3, 4]], [[5, 6]], [[7.8]]])
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[0]], [[None]], [[1], None], [[2, None]]])
    assert (
        to_list(ak.drop_none(a, axis=1))
        == to_list(a[~ak.is_none(a, axis=1)])
        == [[[0]], [[None]], [[1]], [[2, None]]]
    )

    a = ak.Array([[[0]], [None, 34], [[1], None, 31], [[2, [[None]]]], [[[None]]]])
    assert (
        to_list(ak.drop_none(a, axis=0))
        == to_list(a[~ak.is_none(a, axis=0)])
        == [[[0]], [None, 34], [[1], None, 31], [[2, [[None]]]], [[[None]]]]
    )

    a = ak.Array([[[1, None]], [[3, None]], [[5, 6]], [[7.8]]])
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[3.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, 4]], [[5, 6]], [[7.8]]])
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[4.0]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, None]], [[5, 6]], [[7.8]]])
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[]], [[5.0, 6.0]], [[7.8]]]
    )

    a = ak.Array([[[1, None]], [[None, None]], [[None, 6]], [[7.8]]])
    assert (
        to_list(ak.drop_none(a, axis=2))
        == to_list(a[~ak.is_none(a, axis=2)])
        == [[[1.0]], [[]], [[6.0]], [[7.8]]]
    )

    a = ak.Array([[{"x": [1], "y": [[2]]}], [{"x": [None], "y": [[None]]}]])
    assert to_list(a) == [[{"x": [1], "y": [[2]]}], [{"x": [None], "y": [[None]]}]]
    assert to_list(ak.drop_none(a)) == [
        [{"x": [1], "y": [[2]]}],
        [{"x": [], "y": [[]]}],
    ]
    assert to_list(ak.drop_none(a, axis=2)) == [
        [{"x": [1], "y": [[2]]}],
        [{"x": [], "y": [[None]]}],
    ]


def test_List_ByteMaskedArray_NumpyArray():
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([1, 3], np.int64)),
        ak.index.Index(np.array([3, 4], np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=True,
        ),
    )

    assert to_list(a) == [[None, 3.3], [None]]
    assert to_list(ak.drop_none(a)) == [[3.3], []]
    assert to_list(ak.drop_none(a, axis=1)) == to_list(a[~ak.is_none(a, axis=1)])


def test_ListOffsetArray_ByteMaskedArray_NumpyArray():
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            valid_when=True,
        ),
    )

    assert to_list(a) == [[None, 3.3, None], [], [5.5], []]
    assert to_list(ak.drop_none(a, axis=1)) == [[3.3], [], [5.5], []]


def test_ByteMaskedArray_NumpyArray():
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    assert to_list(a) == [1.1, None, 3.3, None, 5.5]
    assert to_list(ak.drop_none(a)) == [1.1, 3.3, 5.5]


def test_BitMaskedArray_NumpyArray():
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert to_list(a) == [
        0.0,
        1.0,
        2.0,
        3.0,
        None,
        None,
        None,
        None,
        1.1,
        None,
        3.3,
        None,
        5.5,
    ]
    assert (
        to_list(ak.drop_none(a))
        == to_list(a[~ak.is_none(a)])
        == [0.0, 1.0, 2.0, 3.0, 1.1, 3.3, 5.5]
    )


def test_UnmaskedArray_NumpyArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )

    assert to_list(ak.drop_none(a)) == [0.0, 1.1, 2.2, 3.3]


def test_BitMaskedArray_RecordArray_NumpyArray():
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert to_list(a) == [
        {"nest": 0.0},
        {"nest": 1.0},
        {"nest": 2.0},
        {"nest": 3.0},
        None,
        None,
        None,
        None,
        {"nest": 1.1},
        None,
        {"nest": 3.3},
        None,
        {"nest": 5.5},
    ]
    assert to_list(ak.drop_none(a)) == [
        {"nest": 0.0},
        {"nest": 1.0},
        {"nest": 2.0},
        {"nest": 3.0},
        {"nest": 1.1},
        {"nest": 3.3},
        {"nest": 5.5},
    ]


def test_RegularArray_RecordArray_NumpyArray():
    index = ak.index.Index64(np.asarray([0, -1, 1, 2, 3, 4, -1, 6, 7, 8, -1, 10]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.listoffsetarray.ListOffsetArray(
        offsets, indexoptionarray
    )
    regulararray = ak.contents.regulararray.RegularArray(listoffsetarray, 2)

    assert to_list(regulararray) == [
        [[0.0, None, 1.1], []],
        [[2.2, 3.3], [4.4]],
        [[None, 6.6, 7.7, 8.8], []],
    ]
    assert to_list(ak.drop_none(regulararray, axis=2)) == [
        [[0.0, 1.1], []],
        [[2.2, 3.3], [4.4]],
        [[6.6, 7.7, 8.8], []],
    ]


def test_IndexedOptionArray_NumpyArray_outoforder():
    index = ak.index.Index64(np.asarray([0, -1, 1, 5, 4, 2, 5]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)

    assert to_list(indexoptionarray) == [0.0, None, 1.1, 5.5, 4.4, 2.2, 5.5]
    assert to_list(ak.drop_none(indexoptionarray)) == [0.0, 1.1, 5.5, 4.4, 2.2, 5.5]


def test_ListOffsetArray_IndexedOptionArray_NumpyArray_outoforder():
    index = ak.index.Index64(np.asarray([0, -1, 1, 5, 4, 2, 5]))
    content = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    offsets = ak.index.Index64(np.asarray([0, 4, 5, 6]))
    listoffset = ak.contents.ListOffsetArray(offsets, indexoptionarray)

    assert to_list(listoffset) == [[0.0, None, 1.1, 5.5], [4.4], [2.2]]
    assert to_list(ak.drop_none(listoffset, axis=1)) == [[0.0, 1.1, 5.5], [4.4], [2.2]]
    assert to_list(ak.drop_none(listoffset)) == [[0.0, 1.1, 5.5], [4.4], [2.2]]


def test_UnionArray_RecordArray_ByteMaskedArray_NumpyArray_noNones():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.contents.recordarray.RecordArray(
                [ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64))],
                ["nest"],
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.bytemaskedarray.ByteMaskedArray(
                        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
                        ak.contents.numpyarray.NumpyArray(
                            np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                        ),
                        valid_when=True,
                    )
                ],
                ["nest"],
            ),
        ],
    )
    assert to_list(v2a) == to_list(ak.drop_none(v2a))


def test_ListArray_IndexedOptionArray_RecordArray_NumpyArray():
    index = ak.index.Index64(np.asarray([0, -1, 1, -1, 4, -1, 5]))
    content = ak.contents.recordarray.RecordArray(
        [ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
        ["nest"],
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        indexoptionarray,
    )
    assert to_list(a) == [
        [{"nest": 4.4}, None, {"nest": 5.5}],
        [],
        [None, {"nest": 1.1}],
    ]
    assert to_list(ak.drop_none(a)) == [
        [{"nest": 4.4}, {"nest": 5.5}],
        [],
        [{"nest": 1.1}],
    ]
    assert to_list(ak.drop_none(a, axis=0)) == [
        [{"nest": 4.4}, None, {"nest": 5.5}],
        [],
        [None, {"nest": 1.1}],
    ]


def test_ListOffsetArray_IndexedOptionArray_RecordArray_NumpyArray():
    index = ak.index.Index64(np.asarray([0, -1, 1, -1, 4, -1, 5]))
    content = ak.contents.recordarray.RecordArray(
        [ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
        ["nest"],
    )
    indexoptionarray = ak.contents.IndexedOptionArray(index, content)
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6], np.int64)),
        indexoptionarray,
    )
    assert to_list(a) == [[None, {"nest": 1.1}, None], [], [{"nest": 4.4}, None]]
    assert (
        to_list(ak.drop_none(a))
        == to_list(ak.drop_none(a, axis=1))
        == [[{"nest": 1.1}], [], [{"nest": 4.4}]]
    )
    assert to_list(ak.drop_none(a, axis=0)) == [
        [None, {"nest": 1.1}, None],
        [],
        [{"nest": 4.4}, None],
    ]


def test_ByteMaskedArray_RecordArray_NumpyArray():
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    assert to_list(a) == [{"nest": 1.1}, None, {"nest": 3.3}, None, {"nest": 5.5}]
    assert to_list(ak.drop_none(a)) == [{"nest": 1.1}, {"nest": 3.3}, {"nest": 5.5}]


def test_highlevel_return():
    assert isinstance(ak.drop_none(ak.Array([1, 2, 3])), ak.Array)
    assert isinstance(ak.drop_none(ak.Array([1, 2, 3]), highlevel=True), ak.Array)
    assert isinstance(
        ak.drop_none(ak.Array([1, 2, 3]), highlevel=False), ak.contents.Content
    )
