# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_array_builder():
    import ctypes

    builder = ak.ArrayBuilder()

    ROOT.gInterpreter.Declare(
        f"""
    #include <functional>

    typedef uint8_t (*FuncPtr)(void*);
    typedef uint8_t (*FuncIntPtr)(void*, int64_t);

    uint8_t
    test_beginlist() {{
        return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    test_endlist() {{
        return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endlist, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    test_integer(int64_t x) {{
        return std::invoke(reinterpret_cast<FuncIntPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_integer, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), x);
    }}
    """
    )
    ROOT.test_beginlist()
    ROOT.test_integer(1)
    ROOT.test_integer(2)
    ROOT.test_integer(3)
    ROOT.test_endlist()

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3]]

    ROOT.test_beginlist()
    ROOT.test_integer(1)
    ROOT.test_integer(2)
    ROOT.test_integer(3)
    ROOT.test_endlist()

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3], [1, 2, 3]]


def test_array_builder_root():
    from awkward._v2._connect.rdataframe.from_rdataframe import connect_ArrayBuilder

    builder = ak.ArrayBuilder()
    func = connect_ArrayBuilder(compiler, builder)

    getattr(ROOT, func["beginlist"])()
    getattr(ROOT, func["integer"])(1)
    getattr(ROOT, func["integer"])(2)
    getattr(ROOT, func["integer"])(3)
    getattr(ROOT, func["endlist"])()
    getattr(ROOT, func["real"])(3.3)

    assert ak.to_list(builder.snapshot()) == [[1, 2, 3], 3.3]


def test_data_frame_vecs():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2")

    ak_array_x = ak._v2.from_rdataframe(
        data_frame_xy, column="x", column_as_record=False
    )
    assert ak_array_x.layout.form == ak._v2.forms.NumpyForm("float64")

    ak_record_array_x = ak._v2.from_rdataframe(
        data_frame_xy, column="x", column_as_record=True
    )
    assert ak_record_array_x.layout.form == ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64")], "x"
    )

    ak_record_array_y = ak._v2.from_rdataframe(
        data_frame_xy, column="y", column_as_record=True
    )
    ak_array = ak._v2.zip([ak_record_array_x, ak_record_array_y])
    assert ak_array.layout.form == ak._v2.forms.RecordForm(
        contents=[
            ak._v2.forms.RecordForm([ak._v2.forms.NumpyForm("float64")], "x"),
            ak._v2.forms.RecordForm([ak._v2.forms.NumpyForm("float64")], "y"),
        ],
        fields=None,
    )


def test_data_frame_rvecs():
    data_frame = ROOT.RDataFrame(1024)
    coordDefineCode = """ROOT::VecOps::RVec<double> {0}(len);
                     std::transform({0}.begin(), {0}.end(), {0}.begin(), [](double){{return gRandom->Uniform(-1.0, 1.0);}});
                     return {0};"""

    data_frame_x_y = (
        data_frame.Define("len", "gRandom->Uniform(0, 16)")
        .Define("x", coordDefineCode.format("x"))
        .Define("y", coordDefineCode.format("y"))
    )

    # Now we have in hands d, a RDataFrame with two columns, x and y, which
    # hold collections of coordinates. The size of these collections vary.
    # Let's now define radii out of x and y. We'll do it treating the collections
    # stored in the columns without looping on the individual elements.
    data_frame_x_y_r = data_frame_x_y.Define("r", "sqrt(x*x + y*y)")
    assert data_frame_x_y_r.GetColumnType("r") == "ROOT::VecOps::RVec<double>"

    array = ak._v2.from_rdataframe(data_frame_x_y_r, column="r", column_as_record=True)

    assert array.layout.form == ak._v2.forms.RecordForm(
        [ak._v2.forms.ListOffsetForm("i64", ak._v2.forms.NumpyForm("float64"))], ["r"]
    )


def test_to_from_data_frame():
    ak_array_in = ak._v2.Array([[1.1], [2.2, 3.3, 4.4], [5.5, 6.6]])
    assert ak_array_in.layout.content.is_contiguous is True

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_out.layout.content.is_contiguous is True

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_to_from_data_frame_rvec_of_rvec():
    ak_array_in = ak._v2.Array([[[1.1]], [[2.2, 3.3], [4.4]], [[5.5, 6.6], []]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_to_from_data_frame_rvec_of_rvec_of_rvec():
    ak_array_in = ak._v2.Array(
        [[[[1.1]]], [[[2.2], [3.3], [], [4.4]]], [[[], [5.5, 6.6], []]]]
    )

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()


@pytest.mark.skip(reason="FIXME: support boolean")
def test_boolean_data_frame():
    ak_array_in = ak._v2.Array([True, True, False, True, False, False])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::RVec<bool>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_complex_vecs():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2 +1i")

    ak_array_y = ak._v2.from_rdataframe(
        data_frame_xy, column="y", column_as_record=False
    )
    assert ak_array_y.layout.form == ak._v2.forms.NumpyForm("complex128")
