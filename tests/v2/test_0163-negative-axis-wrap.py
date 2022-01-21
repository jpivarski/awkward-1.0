# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


def test_array_3d():
    array = ak._v2.highlevel.Array(np.arange(3 * 5 * 2).reshape(3, 5, 2))
    assert to_list(array) == [
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
    ]
    assert ak._v2.operations.structure.num(array, axis=0) == 3
    assert to_list(ak._v2.operations.structure.num(array, axis=1)) == [5, 5, 5]
    assert to_list(ak._v2.operations.structure.num(array, axis=2)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=3)
    assert str(err.value).startswith("axis=3 exceeds the depth of this array (2)")

    assert to_list(ak._v2.operations.structure.num(array, axis=-1)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    assert to_list(ak._v2.operations.structure.num(array, axis=-2)) == [5, 5, 5]
    assert ak._v2.operations.structure.num(array, axis=-3) == 3

    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=-4)
    assert str(err.value).startswith("axis=-4 exceeds the depth (3) of this array")


def test_list_array():
    array = ak._v2.highlevel.Array(np.arange(3 * 5 * 2).reshape(3, 5, 2).tolist())
    assert ak._v2.operations.structure.num(array, axis=0) == 3
    assert ak._v2.operations.structure.num(array, axis=1).tolist() == [5, 5, 5]
    assert ak._v2.operations.structure.num(array, axis=2).tolist() == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]

    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=3)
        assert str(err.value).startswith("axis=3 exceeds the depth of this array (2)")

    assert ak._v2.operations.structure.num(array, axis=-1).tolist() == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    assert ak._v2.operations.structure.num(array, axis=-2).tolist() == [5, 5, 5]
    assert ak._v2.operations.structure.num(array, axis=-3) == 3
    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=-4)
        assert str(err.value).startswith("axis=-4 exceeds the depth of this array (3)")


def test_record_array():
    array = ak._v2.highlevel.Array(
        [
            {"x": [1], "y": [[], [1]]},
            {"x": [1, 2], "y": [[], [1], [1, 2]]},
            {"x": [1, 2, 3], "y": [[], [1], [1, 2], [1, 2, 3]]},
        ]
    )

    assert ak._v2.operations.structure.num(array, axis=0).tolist() == {"x": 3, "y": 3}
    assert ak._v2.operations.structure.num(array, axis=1).tolist() == [
        {"x": 1, "y": 2},
        {"x": 2, "y": 3},
        {"x": 3, "y": 4},
    ]
    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=2)
        assert str(err.value).startswith("axis=2 exceeds the depth of this array (1)")

    assert ak._v2.operations.structure.num(array, axis=-1).tolist() == [
        {"x": 1, "y": [0, 1]},
        {"x": 2, "y": [0, 1, 2]},
        {"x": 3, "y": [0, 1, 2, 3]},
    ]


def test_record_array_axis_out_of_range():
    array = ak._v2.highlevel.Array(
        [
            {"x": [1], "y": [[], [1]]},
            {"x": [1, 2], "y": [[], [1], [1, 2]]},
            {"x": [1, 2, 3], "y": [[], [1], [1, 2], [1, 2, 3]]},
        ]
    )

    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=-2)
        assert str(err.value).startswith("axis=-2 exceeds the depth of this array (2)")

    with pytest.raises(ValueError) as err:
        assert ak._v2.operations.structure.num(array, axis=-3)
        assert str(err.value).startswith("axis=-3 exceeds the depth (2) of this array")
