import pytest  # noqa: F401
import awkward._v2 as ak
from awkward._v2.operations.convert.ak_from_avro import from_avro_file  # noqa: F401


def test_int():
    data = [34, 45, 67, 78, 23, 89, 6, 33, 96, 73]
    assert (
        ak.from_avro_file(
            file="tests/samples/int_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/int_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_boolean():
    data = [True, False, False, True, True, True, False, False, False, False]
    assert (
        ak.from_avro_file(
            file="tests/samples/bool_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/bool_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_long():
    data = [12, 435, 56, 12, 67, 34, 89, 2345, 536, 8769]
    assert (
        ak.from_avro_file(
            file="tests/samples/long_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/long_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_float():
    data = [
        12.456,
        57.1234,
        798.23,
        345.687,
        908.23,
        65.89,
        43.57,
        745.79,
        532.68,
        3387.684,
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/float_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/float_test_data.avro", reader_lang="ft"
            ).to_list()
        )
        == data
    )


def test_double():
    data = [
        12.456,
        57.1234,
        798.23,
        345.687,
        908.23,
        65.89,
        43.57,
        745.79,
        532.68,
        3387.684,
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/double_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/double_test_data.avro", reader_lang="ft"
            ).to_list()
        )
        == data
    )


def test_bytes():
    data = [
        bytes("hello", "utf-8"),
        bytes("hii", "utf-8"),
        bytes("byee", "utf-8"),
        bytes("pink", "utf-8"),
        bytes("blue", "utf-8"),
        bytes("red", "utf-8"),
        bytes("chrome", "utf-8"),
        bytes("green", "utf-8"),
        bytes("black", "utf-8"),
        bytes("peach", "utf-8"),
    ]
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/bytes_test_data.avro", reader_lang="py"
            ).to_list()
        )
        == data
    )
    assert (
        pytest.approx(
            ak.from_avro_file(
                file="tests/samples/bytes_test_data.avro", reader_lang="ft"
            ).to_list()[0]
        )
        == data[0]
    )


def test_string():
    data = [
        "Hello",
        "what",
        "should",
        "we",
        "do",
        "for",
        "this",
        "period",
        "of",
        "time",
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/string_test_data.avro",
            reader_lang="py",
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/string_test_data.avro",
            reader_lang="ft",
        ).to_list()
        == data
    )


def test_fixed():
    data = [
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
        b"like this one",
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/fixed_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/fixed_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_null():  # change the while loop to for loop to fix this
    data = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/null_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/null_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_enum():
    data = ["TWO", "ONE", "FOUR", "THREE", "TWO",
            "ONE", "FOUR", "THREE", "TWO", "ONE"]
    assert (
        ak.from_avro_file(
            file="tests/samples/enum_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/enum_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_arrays_int():
    data = [
        [34, 556, 12],
        [34, 556, 12],
        [34, 532, 657],
        [236, 568, 12],
        [34, 556, 12],
        [34, 54, 967],
        [34, 556, 12],
        [34, 647, 12],
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/array_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/array_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_array_string():
    data = [
        ["afsdfd", "sgrh"],
        ["afsdfd", "sgrh"],
        ["afsdfd", "sgrh"],
        ["afsdfd", "sgrh"],
        ["afsdfd", "sgrh"],
    ]
    assert(ak.from_avro_file(
        file="tests/samples/array_string_test_data.avro", reader_lang="py"
    ).to_list() == data)

    assert(ak.from_avro_file(
        file="tests/samples/array_string_test_data.avro", reader_lang="ft"
    ).to_list() == data)


def test_array_enum():
    data = [
        ["ONE", "FOUR"],
        ["THREE", "ONE"],
        ["THREE", "FOUR"],
        ["TWO", "ONE"],
        ["FOUR", "THREE"],
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/array_enum_test_data.avro", reader_lang="py"
        )[0][0]
        == data[0][0]
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/array_enum_test_data.avro", reader_lang="ft"
        )[0][0]
        == data[0][0]
    )


def test_Unions_int_null():
    data = [2345, 65475, None, 676457, 343, 7908, None, 5768]  # int_null_test
    assert (
        ak.from_avro_file(
            file="tests/samples/int_null_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/int_null_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_Unions_string_null():
    data = ["blue", None, "yellow", None,
            "Green", None, "Red"]  # string_null_test
    assert (
        ak.from_avro_file(
            file="tests/samples/string_null_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/string_null_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_Unions_enum_null():
    data = ["TWO", None, "ONE", None, "FOUR", None, "THREE"]  # enum_null_test
    assert (
        ak.from_avro_file(
            file="tests/samples/enum_null_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/enum_null_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )


def test_Unions_record_null():
    data = [
        {"name": "fegweg", "ex": 2.45},
        None,
        {"name": "therer", "ex": 57.462},
        None,
        {"name": "tedjte", "ex": 653.12},
        None,
    ]

    assert (
        ak.from_avro_file(
            file="tests/samples/record_null_test_data.avro", reader_lang="py"
        ).to_list()[0]["name"]
        == data[0]["name"]
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/record_null_test_data.avro", reader_lang="ft"
        ).to_list()[0]["name"]
        == data[0]["name"]
    )


def test_Unions_null_X_Y():
    data = ["TWO", 5684, "ONE", None, 3154,
            "FOUR", 69645, "THREE"]  # int_string_null
    assert (
        ak.from_avro_file(
            file="tests/samples/int_string_null_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/int_string_null_test_data.avro", reader_lang="ft"
        ).to_list()
        == data
    )
    # assert (
    #    ak.from_avro_file(
    #        file="tests/samples/int_string_null_test_data.avro", reader_lang="py"
    #    ).to_list()
    #    == data
    # )


# def test_record_0():
#    data = []
#    assert (
#        ak.from_avro_file(
#            file="tests/samples/record_0_test_data.avro", reader_lang="py"
#        ).to_list()
#        == data
#    )


def test_record_1():
    data = [
        {"name": "Pierre-Simon Laplace", "age": 77, "Numbers": "TWO"},
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/record_1_test_data.avro", reader_lang="py"
        ).to_list()[0]
        == data[0]
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/record_1_test_data.avro", reader_lang="ft"
        ).to_list()[0]
        == data[0]
    )


def test_records():
    data = [
        {"name": "Pierre-Simon Laplace", "age": 77, "Numbers": "TWO"},
        {"name": "Henry", "age": 36, "Numbers": "THREE"},
        {"name": "Harry", "age": 769, "Numbers": "ONE"},
        {"name": "Jim", "age": 3215, "Numbers": "FOUR"},
        {"name": "Lindsey", "age": 658, "Numbers": "TWO"},
        {"name": "Eduardo", "age": 25, "Numbers": "THREE"},
        {"name": "Aryan", "age": 6478, "Numbers": "FOUR"},
    ]
    assert (
        ak.from_avro_file(
            file="tests/samples/record_test_data.avro", reader_lang="py"
        ).to_list()[0]
        == data[0]
    )
    assert (
        ak.from_avro_file(
            file="tests/samples/record_test_data.avro", reader_lang="ft"
        ).to_list()[0]
        == data[0]
    )
