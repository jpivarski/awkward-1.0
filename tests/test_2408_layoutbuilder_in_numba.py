# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

from numba.core.errors import NumbaTypeError  # noqa: E402

import awkward.numba.layoutbuilder as lb  # noqa: E402

ak.numba.register_and_check()


def test_Numpy():
    builder = lb.Numpy(np.float64)

    builder.append(1.1)
    builder.append(2.2)
    builder.extend([3.3, 4.4, 5.5])

    error = ""
    assert builder.is_valid(error), error

    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.NumpyArray)
    assert str(ak.type(layout)) == "5 * float64"
    assert ak.to_list(layout) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert (
        str(builder.numbatype())
        == "ak.lb.Numpy(float64, parameters=Literal[NoneType](None))"
    )

    assert builder.form == layout.form

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 5
    builder.clear()
    assert len(builder) == 0

    assert builder.is_valid(error), error


def test_Numpy_char():
    builder = lb.Numpy(np.uint8, parameters={"__array__": "char"})
    builder.append(np.uint8(97))
    builder.append(np.uint8(98))
    builder.append(np.uint8(99))

    layout = builder.snapshot()
    assert str(ak.type(layout)) == "3 * char"
    assert ak.to_list(layout) == "abc"  # FIXME: ['a', 'b', 'c']????

    assert (
        str(builder.numbatype())
        == "ak.lb.Numpy(uint8, parameters=Literal[dict]({'__array__': 'char'}))"
    )

    error = ""
    assert builder.is_valid(error), error


def test_Empty():
    builder = lb.Empty()
    assert len(builder) == 0
    assert ak.to_list(builder.snapshot()) == []

    error = ""
    assert builder.is_valid(error), error

    with pytest.raises(AttributeError):
        builder.content.append(1.1)

    with pytest.raises(AttributeError):
        builder.content.extend([3.3, 4.4, 5.5])

    error = ""
    assert builder.is_valid(error), error

    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.EmptyArray)
    assert str(ak.type(layout)) == "0 * unknown"
    assert ak.to_list(layout) == []

    assert str(builder.numbatype()) == "ak.lb.Empty(parameters=Literal[NoneType](None))"

    builder = lb.Empty()
    assert str(builder.numbatype()) == "ak.lb.Empty(parameters=Literal[NoneType](None))"
    assert builder.form == layout.form


def test_ListOffset():
    builder = lb.ListOffset(np.int64, lb.Numpy(np.float64))  # FIXME: np.int32
    assert len(builder) == 0
    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.to_list(layout) == []

    content = builder.begin_list()
    content.append(1.1)
    content.append(2.2)
    content.append(3.3)
    builder.end_list()

    builder.begin_list()
    builder.end_list()

    builder.begin_list()
    content.append(4.4)
    content.append(5.5)
    builder.end_list()

    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.to_list(layout) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert builder.form == layout.form

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.ListOffset(int64, ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_Regular():
    builder = lb.Regular(lb.Numpy(np.float64), 3)
    assert len(builder) == 0
    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.RegularArray)
    assert ak.to_list(layout) == []

    content = builder.begin_list()
    content.append(1.1)
    content.append(2.2)
    content.append(3.3)
    builder.end_list()

    builder.begin_list()
    content.append(4.4)
    content.append(5.5)
    content.append(6.6)
    builder.end_list()

    layout = builder.snapshot()
    assert ak.to_list(layout) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    assert len(builder) == 2
    assert builder.form == layout.form

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.Regular(ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), 3, parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_Regular_size0():
    with pytest.raises(ValueError):
        lb.Regular(lb.Numpy(np.float64), 0)


def test_IndexedOption():
    builder = lb.IndexedOption(np.int64, lb.Numpy(np.float64))
    assert len(builder) == 0

    content = builder.append_valid()
    content.append(1.1)
    builder.append_invalid()

    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    builder.extend_valid(len(data))
    content.extend(data)

    builder.extend_invalid(2)
    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, None, 3.3, 4.4, 5.5, None, None]
    assert builder.form == layout.form

    assert len(builder) == 7

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.IndexedOption(int64, ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_Record():
    builder = lb.Record(
        [
            lb.Numpy(np.float64),
            lb.Numpy(np.int64),
            lb.Numpy(np.uint8, parameters={"__array__": "char"}),
        ],
        ["one", "two", "three"],
    )
    assert len(builder) == 0

    one = builder.content("one")
    two = builder.content("two")
    three = builder.content("three")

    three.append(np.uint8(0x61))  #'a')

    one.append(1.1)
    one.append(3.3)

    two.append(2)
    two.append(4)

    three.append(np.uint8(0x62))  #'b')

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        {"one": 1.1, "two": 2, "three": "a"},
        {"one": 3.3, "two": 4, "three": "b"},
    ]
    assert builder.form == layout.form

    assert len(builder) == 2

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.Record((ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), ak.lb.Numpy(int64, parameters=Literal[NoneType](None)), ak.lb.Numpy(uint8, parameters=Literal[dict]({'__array__': 'char'}))), ('one', 'two', 'three'), parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_IndexedOption_Record():
    builder = lb.IndexedOption(
        np.int64, lb.Record([lb.Numpy(np.float64), lb.Numpy(np.int64)], ["x", "y"])
    )
    assert len(builder) == 0
    record = builder.append_valid()
    x = record.content("x")
    y = record.content("y")

    x.append(1.1)
    y.append(2)

    builder.append_invalid()

    builder.append_valid()
    x.append(3.3)
    y.append(4)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        {"x": 1.1, "y": 2},
        None,
        {"x": 3.3, "y": 4},
    ]
    assert builder.form == layout.form


def test_Tuple_Numpy_ListOffset():
    builder = lb.Tuple(
        [lb.Numpy(np.float64), lb.ListOffset(np.int64, lb.Numpy(np.int32))]
    )
    assert len(builder) == 0

    error = ""
    assert builder.is_valid(error) is True

    one = builder.index(0)
    one.append(1.1)
    two = builder.index(1)
    two_list = two.begin_list()
    two_list.append(np.int32(1))
    two.end_list()

    assert builder.is_valid(error) is True

    one.append(2.2)
    two.begin_list()
    two_list.append(np.int32(1))
    two_list.append(np.int32(2))
    two.end_list()

    assert builder.is_valid(error) is True

    one.append(3.3)
    two.begin_list()
    two_list.append(np.int32(1))
    two_list.append(np.int32(2))
    two_list.append(np.int32(3))
    two.end_list()

    layout = builder.snapshot()
    assert ak.to_list(layout) == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]

    assert builder.is_valid(error) is True
    assert builder.form == layout.form


def test_Unmasked():
    builder = lb.Unmasked(lb.Numpy(np.int64))
    assert len(builder) == 0

    content = builder._content
    content.append(11)
    content.append(22)
    content.append(33)
    content.append(44)
    content.append(55)

    err = ""
    assert builder.is_valid(err) is True

    layout = builder.snapshot()
    assert ak.to_list(layout) == [11, 22, 33, 44, 55]
    assert builder.form == layout.form

    assert (
        str(builder.numbatype())
        == "ak.lb.Unmasked(ak.lb.Numpy(int64, parameters=Literal[NoneType](None)), parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_ByteMasked():
    builder = lb.ByteMasked(lb.Numpy(np.float64), valid_when=True)
    assert len(builder) == 0

    content = builder.append_valid()
    content.append(1.1)

    builder.append_invalid()
    content.append(np.nan)

    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    builder.extend_valid(3)
    content.extend(data)

    builder.extend_invalid(2)
    content.append(np.nan)
    content.append(np.nan)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, None, 3.3, 4.4, 5.5, None, None]
    assert builder.form == layout.form

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.ByteMasked(ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), valid_when=True, parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_BitMasked():
    builder = lb.BitMasked(np.uint8, lb.Numpy(np.float64), True, True)
    assert len(builder) == 0

    subbuilder = builder.append_valid()
    subbuilder.append(1.1)
    assert len(builder) == 1
    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1]
    assert builder.form == layout.form

    builder.append_invalid()
    subbuilder.append(np.nan)
    assert len(builder) == 2

    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)

    builder.extend_valid(3)
    subbuilder.extend(data)
    assert len(builder) == 5

    builder.extend_invalid(2)
    subbuilder.append(np.nan)
    subbuilder.append(np.nan)

    assert len(builder) == 7

    builder.append_valid()
    subbuilder.append(8)
    assert len(builder) == 8

    builder.append_valid()
    subbuilder.append(9)
    assert len(builder) == 9

    builder.append_valid()
    subbuilder.append(10)
    assert len(builder) == 10

    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, None, 3.3, 4.4, 5.5, None, None, 8, 9, 10]

    error = ""
    assert builder.is_valid(error), error

    assert (
        str(builder.numbatype())
        == "ak.lb.BitMasked(uint8, ak.lb.Numpy(float64, parameters=Literal[NoneType](None)), True, True, parameters=Literal[NoneType](None))"
    )
    builder.clear()
    assert len(builder) == 0


def test_Union_Numpy_ListOffset():
    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.Numpy(np.float64),
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
        ],
    )
    assert len(builder) == 0

    # error = ""
    # assert builder.is_valid(error) == True

    one = builder.append_content(0)
    one.append(1.1)

    # assert builder.is_valid(error) == True

    two = builder.append_content(1)
    list = two.begin_list()
    list.append(np.int32(1))
    list.append(np.int32(2))
    two.end_list()

    # assert builder.is_valid(error) == True

    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, [1, 2]]
    assert builder.form == layout.form


def test_Union_ListOffset_Record():
    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
            lb.Record([lb.Numpy(np.float64), lb.Numpy(np.int64)], ["x", "y"]),
        ],
    )
    assert len(builder) == 0

    one = builder.append_content(0)
    list = one.begin_list()
    list.append(1)
    list.append(3)
    one.end_list()

    two = builder.append_content(1)
    x = two.content("x")
    y = two.content("y")

    x.append(1.1)
    y.append(11)

    builder.append_content(0)
    list = one.begin_list()
    list.append(5.5)
    one.end_list()

    builder.append_content(1)
    x.append(2.2)
    y.append(22)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [[1, 3], {"x": 1.1, "y": 11}, [5], {"x": 2.2, "y": 22}]
    assert builder.form == layout.form


def test_unbox():
    @numba.njit(debug=True)
    def f1(x):
        x  # noqa: B018 (we want to test the unboxing)
        return 3.14

    builder = lb.BitMasked(np.uint8, lb.Numpy(np.float64), True, True)
    f1(builder)

    builder = lb.ByteMasked(lb.Numpy(np.float64), valid_when=True)
    f1(builder)

    builder = lb.Empty()
    f1(builder)

    builder = lb.IndexedOption(np.int64, lb.Numpy(np.float64))
    f1(builder)

    builder = lb.IndexedOption(np.int32, lb.Numpy(np.float64))
    f1(builder)

    builder = lb.ListOffset(np.int32, lb.ListOffset(np.int64, lb.Numpy(np.int64)))
    f1(builder)

    builder = lb.ListOffset(np.int32, lb.Numpy(np.float64))
    f1(builder)

    builder = lb.ListOffset(np.int32, lb.Empty())
    f1(builder)

    builder = lb.ListOffset(np.int8, lb.Numpy(np.int64))
    f1(builder)

    builder = lb.ListOffset(np.int32, lb.Empty())
    f1(builder)

    builder = lb.Numpy(np.int32)
    f1(builder)

    builder = lb.Record(
        [
            lb.Numpy(np.float64),
            lb.Numpy(np.int64),
            lb.Numpy(np.uint8),
        ],
        ["one", "two", "three"],
    )
    f1(builder)

    @numba.njit(debug=True)
    def f_compiled_index(x):
        i = x._contents[0]
        i = x._contents[1]
        i = x._contents[2]
        return i

    f_compiled_index(builder)

    @numba.njit(debug=True)
    def f_runtime_index(x, name):
        return x.content("one")  # name) # FIXME: numba.types.UnicodeType

    content = f_runtime_index(builder, "one")
    assert content.numbatype() == builder._contents[0].numbatype()

    builder = lb.Regular(lb.Numpy(np.float64), size=3)
    f1(builder)

    builder = lb.Tuple(
        [lb.Numpy(np.float64), lb.ListOffset(np.int64, lb.Numpy(np.int32))]
    )
    f1(builder)

    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.Numpy(np.float64),
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
        ],
    )
    f1(builder)

    builder = lb.Unmasked(lb.Numpy(np.int64))
    f1(builder)


def test_unbox_for_loop():
    @numba.njit
    def f2(x):
        for i in range(10):
            x.append(i)
        return

    builder = lb.Numpy(np.int64, parameters=None)
    f2(builder)
    assert ak.to_list(builder.snapshot()) == list(range(10))

    builder = lb.Empty()
    # Unknown attribute 'append' of type lb.Empty
    with pytest.raises(NumbaTypeError):
        f2(builder)


def test_box():
    @numba.njit
    def f3(x):
        return x

    builder = lb.Numpy(np.int32)
    out1 = f3(builder)
    assert ak.to_list(out1.snapshot()) == []

    for x in range(15):
        out1.append(np.int32(x))

    out2 = f3(out1)
    assert ak.to_list(out2.snapshot()) == list(range(15))

    builder = lb.Empty()
    out3 = f3(builder)
    assert ak.to_list(out3.snapshot()) == []

    builder = lb.ListOffset(np.int64, lb.Numpy(np.int32))
    out4 = f3(builder)
    assert ak.to_list(out4.snapshot()) == []

    builder = lb.ListOffset(np.int64, lb.Numpy(np.int64))
    out5 = f3(builder)
    assert ak.to_list(out5.snapshot()) == []

    builder = lb.ListOffset(np.int32, lb.Empty())
    out6 = f3(builder)
    assert ak.to_list(out6.snapshot()) == []

    builder = lb.ListOffset(np.int32, lb.ListOffset(np.int64, lb.Numpy(np.int64)))
    out5 = f3(builder)
    assert ak.to_list(out5.snapshot()) == []

    builder = lb.Regular(lb.Numpy(np.float64), 3)
    out7 = f3(builder)
    assert ak.to_list(out7.snapshot()) == []

    builder = lb.ListOffset(np.int32, lb.Regular(lb.Numpy(np.float64), 3))
    out8 = f3(builder)
    assert ak.to_list(out8.snapshot()) == []

    builder = lb.IndexedOption(np.int32, lb.Numpy(np.float64))
    out9 = f3(builder)
    assert ak.to_list(out9.snapshot()) == []

    builder = lb.IndexedOption(np.int64, lb.Numpy(np.float64))
    out10 = f3(builder)
    assert ak.to_list(out10.snapshot()) == []

    builder = lb.ByteMasked(lb.Numpy(np.float64), valid_when=True)
    out11 = f3(builder)
    assert ak.to_list(out11.snapshot()) == []

    builder = lb.BitMasked(np.uint8, lb.Numpy(np.float64), True, True)
    out12 = f3(builder)
    assert ak.to_list(out12.snapshot()) == []

    builder = lb.Unmasked(lb.Numpy(np.int64))
    out13 = f3(builder)
    assert ak.to_list(out13.snapshot()) == []

    builder = lb.Record(
        [
            lb.Numpy(np.float64),
            lb.Numpy(np.int64),
            lb.Numpy(np.uint8),
        ],
        ["one", "two", "three"],
    )
    out14 = f3(builder)
    assert ak.to_list(out14.snapshot()) == []

    builder = lb.Tuple(
        [lb.Numpy(np.float64), lb.ListOffset(np.int64, lb.Numpy(np.int32))]
    )
    out15 = f3(builder)
    assert ak.to_list(out15.snapshot()) == []

    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.Numpy(np.float64),
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
        ],
    )
    out16 = f3(builder)
    assert ak.to_list(out16.snapshot()) == []


def test_len():
    @numba.njit
    def f4(x):
        return len(x)

    builder = lb.Numpy(np.int32, parameters=None)
    assert f4(builder) == 0

    builder.append(np.int32(123))
    assert f4(builder) == 1

    builder = lb.Empty()
    assert f4(builder) == 0

    builder = lb.ListOffset(np.int64, lb.Numpy(np.int32))
    assert f4(builder) == 0

    builder = lb.ListOffset(np.int8, lb.Empty())
    assert f4(builder) == 0

    builder = lb.ListOffset(np.int32, lb.ListOffset(np.int32, lb.Numpy(np.int64)))
    assert f4(builder) == 0

    builder = lb.ListOffset(np.int64, lb.Numpy(np.int32))
    assert f4(builder) == 0

    builder = lb.Regular(lb.Numpy(np.float64), 3)
    assert f4(builder) == 0

    builder = lb.IndexedOption(np.int32, lb.Numpy(np.float64))
    assert f4(builder) == 0

    builder = lb.IndexedOption(np.int64, lb.Numpy(np.float64))
    assert f4(builder) == 0

    builder = lb.ByteMasked(lb.Numpy(np.float64), valid_when=True)
    assert f4(builder) == 0

    builder = lb.BitMasked(np.uint8, lb.Numpy(np.float64), True, True)
    assert f4(builder) == 0

    builder = lb.Unmasked(lb.Numpy(np.int64))
    assert f4(builder) == 0

    builder = lb.Record(
        [
            lb.Numpy(np.float64),
            lb.Numpy(np.int64),
            lb.Numpy(np.uint8),
        ],
        ["one", "two", "three"],
    )
    assert f4(builder) == 0

    builder = lb.Tuple(
        [lb.Numpy(np.float64), lb.ListOffset(np.int64, lb.Numpy(np.int32))]
    )
    assert f4(builder) == 0

    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.Numpy(np.float64),
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
        ],
    )
    assert f4(builder) == 0


# def test_Numpy_from_buffer():
#     @numba.njit
#     def f5(debug=True):
#         growablebuffer = ak.numba.GrowableBuffer(numba.float64)
#         growablebuffer.append(66.6)
#         growablebuffer.append(77.7)
#         return growablebuffer
#
#     out = f5()
#     assert out.snapshot().tolist() == [66.6, 77.7]
#
#     @numba.njit
#     def f6():
#         growablebuffer = ak.numba.GrowableBuffer(numba.float64)
#         growablebuffer.append(66.6)
#         growablebuffer.append(77.7)
#
#         return ak._connect.numba.layoutbuilder._from_buffer(growablebuffer)
#
#     out = f6()
#     assert isinstance(out, lb.Numpy)
#     assert out.dtype == np.dtype(np.float64)
#     assert len(out) == 2
#
#     assert ak.to_list(out.snapshot()) == [66.6, 77.7]


def test_Numpy_ctor():
    # @numba.njit
    # def f7():
    #     return lb.Numpy("f4")
    #
    # out = f7()
    # assert isinstance(out, lb.Numpy)
    # assert out.dtype == np.dtype("f4")
    # assert len(out) == 0

    @numba.njit
    def f8():
        return lb.Numpy(np.float32)

    out = f8()
    assert isinstance(out, lb.Numpy)
    assert out.dtype == np.float32
    assert len(out) == 0

    @numba.njit
    def f9():
        return lb.Numpy(np.dtype(np.float32))

    out = f9()
    assert isinstance(out, lb.Numpy)
    assert out.dtype == np.dtype(np.float32)
    assert len(out) == 0


def test_Numpy_append():
    @numba.njit
    def f10(builder):
        for i in range(8):
            builder.append(np.float32(i))

    builder = lb.Numpy(np.float32)

    f10(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8))

    f10(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8)) + list(range(8))


def test_Numpy_extend():
    @numba.njit
    def f11(builder):
        builder.extend(np.arange(8, dtype=np.float32))

    builder = lb.Numpy(np.float32)

    f11(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8))

    f11(builder)

    assert ak.to_list(builder.snapshot()) == list(range(8)) + list(range(8))


def test_ListOffset_begin_list():
    @numba.njit
    def f13(builder):
        return builder.begin_list()

    builder = lb.ListOffset(np.int64, lb.Numpy(np.int32))

    out = f13(builder)
    assert isinstance(out, lb.Numpy)
    assert out.dtype == np.int32


def test_ListOffset_end_list():
    @numba.njit
    def f14(builder):
        builder.begin_list()
        builder.end_list()

        builder.begin_list()
        builder.end_list()

        builder.begin_list()
        builder.end_list()

    builder = lb.ListOffset(np.int64, lb.Numpy(np.float64))
    assert len(builder) == 0

    f14(builder)
    assert len(builder) == 3

    assert ak.to_list(builder.snapshot()) == [[], [], []]


def test_ListOffset_append():
    @numba.njit
    def f15(builder):
        content = builder.begin_list()
        content.append(1.1)
        content.append(2.2)
        content.append(3.3)
        builder.end_list()

        builder.begin_list()
        builder.end_list()

        builder.begin_list()
        content.append(4.4)
        content.append(5.5)
        builder.end_list()

    builder = lb.ListOffset(np.int64, lb.Numpy(np.float64))
    assert len(builder) == 0

    f15(builder)
    assert len(builder) == 3

    assert ak.to_list(builder.snapshot()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_ListOffset_as_string():
    @numba.njit
    def f16(builder):
        content = builder.begin_list()
        content.append(np.uint8(104))  # 'h'
        content.append(np.uint8(101))  # 'e'
        content.append(np.uint8(108))  # 'l'
        content.append(np.uint8(108))  # 'l'
        content.append(np.uint8(111))  # 'o'
        builder.end_list()

        builder.begin_list()
        content.append(np.uint8(119))  # 'w'
        content.append(np.uint8(111))  # 'o'
        content.append(np.uint8(114))  # 'r'
        content.append(np.uint8(108))  # 'l'
        content.append(np.uint8(100))  # 'd'
        builder.end_list()

    builder = lb.ListOffset(
        np.int64,
        lb.Numpy(np.uint8, parameters={"__array__": "char"}),
        parameters={"__array__": "string"},
    )
    f16(builder)

    assert ak.to_list(builder.snapshot()) == ["hello", "world"]


def test_ListOffset_append2():
    @numba.njit
    def f17(builder):
        content = builder.begin_list()
        content.append(1.1)
        content.append(2.2)
        content.append(3.3)
        builder.end_list()

        builder.begin_list()
        builder.end_list()

        builder.begin_list()
        content.append(4.4)
        content.append(5.5)
        builder.end_list()

    builder = lb.ListOffset(np.int32, lb.Numpy(np.float64))
    assert len(builder) == 0

    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.to_list(layout) == []

    f17(builder)

    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.to_list(layout) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 3
    builder.clear()
    assert len(builder) == 0


def test_Regular_append():
    @numba.njit
    def f18(builder):
        content = builder.begin_list()
        content.append(1.1)
        content.append(2.2)
        content.append(3.3)
        builder.end_list()

        builder.begin_list()
        content.append(4.4)
        content.append(5.5)
        content.append(6.6)
        builder.end_list()

    builder = lb.Regular(lb.Numpy(np.float64), 3)
    assert len(builder) == 0
    layout = builder.snapshot()
    assert isinstance(layout, ak.contents.RegularArray)
    assert ak.to_list(layout) == []

    f18(builder)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    error = ""
    assert builder.is_valid(error), error


def test_IndexedOption_Record_append():
    @numba.njit
    def f19(builder):
        record = builder.append_valid()
        x = record.content("x")
        y = record.content("y")
        x.append(1.1)
        y.append(2)
        builder.append_invalid()

        builder.append_valid()
        x.append(3.3)
        y.append(4)

    builder = lb.IndexedOption(
        np.int64, lb.Record([lb.Numpy(np.float64), lb.Numpy(np.int64)], ["x", "y"])
    )
    assert len(builder) == 0

    f19(builder)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        {"x": 1.1, "y": 2},
        None,
        {"x": 3.3, "y": 4},
    ]
    assert builder.form == layout.form


def test_IndexedOption_append_extend():
    @numba.njit
    def f21(builder):
        content = builder.append_valid()
        content.append(1.1)
        builder.append_valid()
        content.append(2.2)
        builder.append_valid()
        content.append(3.3)
        builder.append_valid()
        content.append(4.4)
        builder.append_valid()
        content.append(5.5)
        builder.append_valid()
        content.append(6.6)

    @numba.njit
    def f22(builder):
        builder.append_invalid()

    @numba.njit
    def f23(builder, data):
        content = builder.extend_valid(len(data))
        content.extend(data)

    @numba.njit
    def f24(builder, size):
        builder.extend_invalid(size)

    builder = lb.IndexedOption(np.int64, lb.Numpy(np.float64))
    assert len(builder) == 0

    f21(builder)
    assert len(builder) == 6

    f22(builder)
    assert len(builder) == 7

    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, None]

    # FIXME: keep track of self._last_valid
    # error = ""
    # assert builder.is_valid(error), error

    data = np.array([33.33, 44.44, 55.55], dtype=np.float64)
    f23(builder, data)
    assert len(builder) == 10

    f24(builder, 3)
    assert len(builder) == 13

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        None,
        33.33,
        44.44,
        55.55,
        None,
        None,
        None,
    ]

    # FIXME: assert builder.is_valid(error), error


def test_ByteMasked_append_extend():
    @numba.njit
    def f25(builder):
        content = builder.append_valid()
        content.append(1.1)
        builder.append_valid()
        content.append(2.2)
        builder.append_valid()
        content.append(3.3)
        builder.append_valid()
        content.append(4.4)
        builder.append_valid()
        content.append(5.5)
        builder.append_valid()
        content.append(6.6)

    @numba.njit
    def f26(builder):
        content = builder.append_invalid()
        content.append(np.nan)

    @numba.njit
    def f27(builder, data):
        content = builder.extend_valid(len(data))
        content.extend(data)

    @numba.njit
    def f28(builder, size):
        content = builder.extend_invalid(size)
        content.extend([np.nan] * size)

    builder = lb.ByteMasked(lb.Numpy(np.float64), valid_when=True)
    assert len(builder) == 0

    f25(builder)
    f26(builder)
    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    f27(builder, data)

    f28(builder, 2)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        None,
        3.3,
        4.4,
        5.5,
        None,
        None,
    ]

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 12
    builder.clear()
    assert len(builder) == 0


def test_BitMasked_append_extend():
    @numba.njit
    def f29(builder):
        content = builder.append_valid()
        content.append(1.1)
        builder.append_valid()
        content.append(2.2)
        builder.append_valid()
        content.append(3.3)
        builder.append_valid()
        content.append(4.4)
        builder.append_valid()
        content.append(5.5)
        builder.append_valid()
        content.append(6.6)

    @numba.njit
    def f30(builder):
        content = builder.append_invalid()
        content.append(np.nan)

    @numba.njit
    def f31(builder, data):
        content = builder.extend_valid(len(data))
        content.extend(data)

    @numba.njit
    def f32(builder, size):
        content = builder.extend_invalid(size)
        content.append(np.nan)
        content.append(np.nan)

    builder = lb.BitMasked(np.uint8, lb.Numpy(np.float64), True, True)
    assert len(builder) == 0

    f29(builder)
    f30(builder)
    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    f31(builder, data)

    f32(builder, 2)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        None,
        3.3,
        4.4,
        5.5,
        None,
        None,
    ]

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 12
    builder.clear()
    assert len(builder) == 0


def test_Unmasked_append_extend():
    @numba.njit
    def f33(builder):
        content = builder._content
        content.append(1.1)
        content.append(2.2)
        content.append(3.3)
        content.append(4.4)
        content.append(5.5)
        content.append(6.6)

    @numba.njit
    def f34(builder, data):
        content = builder._content
        content.extend(data)

    builder = lb.Unmasked(lb.Numpy(np.float64))

    f33(builder)
    data = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    f34(builder, data)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        3.3,
        4.4,
        5.5,
    ]

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 9
    builder.clear()
    assert len(builder) == 0


def test_Record_content():
    @numba.njit
    def field_index(builder, name):
        return builder._field_index(name)

    @numba.njit
    def fill(builder):
        content_one = builder.content("one")
        content_one.append(1.1)
        content_two = builder.content("two")
        content_two.append(1)
        content_three = builder.content("three")
        content_three.append(np.uint8(111))

    builder = lb.Record(
        [
            lb.Numpy(np.float64),
            lb.Numpy(np.int64),
            lb.Numpy(np.uint8, parameters={"__array__": "char"}),
        ],
        ["one", "two", "three"],
    )
    assert field_index(builder, "one") == 0
    assert field_index(builder, "two") == 1
    assert field_index(builder, "three") == 2

    with pytest.raises(ValueError):
        field_index(builder, "four")  # ValueError: tuple.index(x): x not in tuple

    fill(builder)
    layout = builder.snapshot()
    assert ak.to_list(layout) == [{"one": 1.1, "three": "o", "two": 1}]  # ???
    assert builder.form == layout.form

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 1
    builder.clear()
    assert len(builder) == 0


def test_Tuple_append():
    @numba.njit
    def f38(builder):
        content_one = builder.content(0)
        content_one.append(1.1)
        content_two = builder.content(1)
        content_list = content_two.begin_list()
        content_list.append(1)
        content_list.append(2)
        content_list.append(3)
        content_two.end_list()

    builder = lb.Tuple(
        [lb.Numpy(np.float64), lb.ListOffset(np.int64, lb.Numpy(np.int32))]
    )
    f38(builder)
    layout = builder.snapshot()
    assert ak.to_list(layout) == [(1.1, [1, 2, 3])]

    error = ""
    assert builder.is_valid(error), error

    assert len(builder) == 1
    builder.clear()
    assert len(builder) == 0


def test_Union_append():
    @numba.njit
    def f39(builder):
        one = builder.append_content(0)
        one.append(1.1)

        two = builder.append_content(1)
        list = two.begin_list()
        list.append(np.int32(1))
        list.append(np.int32(2))
        list.append(np.int32(3))
        two.end_list()

    builder = lb.Union(
        np.int8,
        np.int64,
        [
            lb.Numpy(np.float64),
            lb.ListOffset(np.int64, lb.Numpy(np.int32)),
        ],
    )
    assert len(builder) == 0

    f39(builder)

    layout = builder.snapshot()
    assert ak.to_list(layout) == [1.1, [1, 2, 3]]


def test_numba_append():
    @numba.njit
    def create():
        return lb.Numpy(np.int32)

    @numba.njit
    def append_range(builder, start, stop):
        for x in range(start, stop):
            builder.append(np.int32(x))  # note cast!

    @numba.njit
    def append_single(builder, x):
        builder.append(np.int32(x))  # note cast!

    builder = create()
    assert ak.to_list(builder.snapshot()) == []
    assert len(builder) == 0

    append_range(builder, 0, 5)
    assert ak.to_list(builder.snapshot()) == list(range(5))
    assert len(builder) == 5

    append_range(builder, 5, 9)
    assert ak.to_list(builder.snapshot()) == list(range(9))
    assert len(builder) == 9

    append_single(builder, 9)
    assert ak.to_list(builder.snapshot()) == list(range(10))
    assert len(builder) == 10

    append_single(builder, 10)
    assert ak.to_list(builder.snapshot()) == list(range(11))
    assert len(builder) == 11

    append_single(builder, 11)
    assert ak.to_list(builder.snapshot()) == list(range(12))
    assert len(builder) == 12

    append_range(builder, 12, 30)
    assert ak.to_list(builder.snapshot()) == list(range(30))
    assert len(builder) == 30

    append_single(builder, 30)
    assert ak.to_list(builder.snapshot()) == list(range(31))
    assert len(builder) == 31
