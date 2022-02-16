# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


@pytest.mark.skip(
    reason="FIXME:  setitem changes in v2"
)
def test_record():
    array1 = ak._v2.operations.convert.from_iter(
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], highlevel=False
    )
    assert to_list(array1) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]

    array2 = array1.setitem_field(
        "z", ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array2) == [
        {"x": 1, "y": 1.1, "z": []},
        {"x": 2, "y": 2.2, "z": [1]},
        {"x": 3, "y": 3.3, "z": [2, 2]},
    ]

    array3 = array1.setitem_field(
        None, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [
        {"x": 1, "y": 1.1, "2": []},
        {"x": 2, "y": 2.2, "2": [1]},
        {"x": 3, "y": 3.3, "2": [2, 2]},
    ]

    array3 = array1.setitem_field(
        0, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [
        {"x": 1, "y": 1.1, "0": []},
        {"x": 2, "y": 2.2, "0": [1]},
        {"x": 3, "y": 3.3, "0": [2, 2]},
    ]

    array1 = ak._v2.operations.convert.from_iter(
        [(1, 1.1), (2, 2.2), (3, 3.3)], highlevel=False
    )
    assert to_list(array1) == [(1, 1.1), (2, 2.2), (3, 3.3)]

    array2 = array1.setitem_field(
        "z", ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array2) == [
        {"0": 1, "1": 1.1, "z": []},
        {"0": 2, "1": 2.2, "z": [1]},
        {"0": 3, "1": 3.3, "z": [2, 2]},
    ]

    array3 = array1.setitem_field(
        None, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

    array3 = array1.setitem_field(
        0, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [([], 1, 1.1), ([1], 2, 2.2), ([2, 2], 3, 3.3)]

    array3 = array1.setitem_field(
        1, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [(1, [], 1.1), (2, [1], 2.2), (3, [2, 2], 3.3)]

    array3 = array1.setitem_field(
        100, ak._v2.operations.convert.from_iter([[], [1], [2, 2]], highlevel=False)
    )
    assert to_list(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]


def test_withfield():
    base = ak._v2.Array([{"x": 1}, {"x": 2}, {"x": 3}], check_valid=True)
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(ak._v2.operations.structure.with_field(base, what)) == [
        {"x": 1, "1": 1.1},
        {"x": 2, "1": 2.2},
        {"x": 3, "1": 3.3},
    ]
    assert to_list(ak._v2.operations.structure.with_field(base, what, where="y")) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]

    base["z"] = what
    assert to_list(base) == [
        {"x": 1, "z": 1.1},
        {"x": 2, "z": 2.2},
        {"x": 3, "z": 3.3},
    ]

    base["q"] = 123
    assert to_list(base) == [
        {"x": 1, "z": 1.1, "q": 123},
        {"x": 2, "z": 2.2, "q": 123},
        {"x": 3, "z": 3.3, "q": 123},
    ]

    base = ak._v2.Array([{"x": 1}, {"x": 2}, {"x": 3}], check_valid=True)[2]
    assert to_list(ak._v2.operations.structure.with_field(base, 100, "y")) == {
        "x": 3,
        "y": 100,
    }


def test_regulararray():
    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    recordarray = ak._v2.contents.RecordArray([content], fields=["x"])
    regulararray = ak._v2.Array(
        ak._v2.contents.RegularArray(recordarray, 3, zeros_length=0), check_valid=True
    )

    content2 = ak._v2.contents.NumpyArray(np.array([100, 200, 300]))
    regulararray2 = ak._v2.Array(
        ak._v2.contents.RegularArray(content2, 1, zeros_length=0), check_valid=True
    )
    assert to_list(
        ak._v2.operations.structure.with_field(regulararray, regulararray2, "y")
    ) == [
        [{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}],
        [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}],
        [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}],
    ]

    content2 = ak._v2.contents.NumpyArray(
        np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    )
    regulararray2 = ak._v2.Array(
        ak._v2.contents.RegularArray(content2, 3, zeros_length=0), check_valid=True
    )
    assert to_list(
        ak._v2.operations.structure.with_field(regulararray, regulararray2, "y")
    ) == [
        [{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}],
        [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}],
        [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}],
    ]

    content2 = ak._v2.Array(
        ak._v2.contents.NumpyArray(np.array([[100], [200], [300]])), check_valid=True
    )
    assert to_list(
        ak._v2.operations.structure.with_field(regulararray, content2, "y")
    ) == [
        [{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}],
        [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}],
        [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}],
    ]

    content2 = ak._v2.Array(
        ak._v2.contents.NumpyArray(
            np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])
        ),
        check_valid=True,
    )
    assert to_list(
        ak._v2.operations.structure.with_field(regulararray, content2, "y")
    ) == [
        [{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}],
        [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}],
        [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}],
    ]


def test_listarray():
    one = ak._v2.Array(
        [[{"x": 1}, {"x": 2}, {"x": 3}], [], [{"x": 4}, {"x": 5}]], check_valid=True
    )
    two = ak._v2.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    assert to_list(ak._v2.operations.structure.with_field(one, two, "y")) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ]

    three = ak._v2.Array([100, 200, 300], check_valid=True)
    assert to_list(ak._v2.operations.structure.with_field(one, three, "y")) == [
        [{"x": 1, "y": 100}, {"x": 2, "y": 100}, {"x": 3, "y": 100}],
        [],
        [{"x": 4, "y": 300}, {"x": 5, "y": 300}],
    ]

    assert to_list(
        ak._v2.operations.structure.with_field(one, [100, 200, 300], "y")
    ) == [
        [{"x": 1, "y": 100}, {"x": 2, "y": 100}, {"x": 3, "y": 100}],
        [],
        [{"x": 4, "y": 300}, {"x": 5, "y": 300}],
    ]
