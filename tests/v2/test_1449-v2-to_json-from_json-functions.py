# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import base64

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_to_json_options(tmp_path):
    filename = os.path.join(tmp_path, "whatever.json")

    array = ak._v2.Array(
        [
            {"x": 1.1, "y": 1 + 1j, "z": b"one"},
            {"x": 2.2, "y": 2 + 2j, "z": b"two"},
            {"x": 3.3, "y": 3 + 3j, "z": b"three"},
            {"x": float("nan"), "y": float("nan"), "z": b"four"},
            {"x": float("inf"), "y": float("inf") + 5j, "z": b"five"},
            {"x": float("-inf"), "y": 6 + float("-inf") * 1j, "z": b"six"},
            {"x": 7.7, "y": 7 + 7j, "z": b"seven"},
            {"x": None, "y": 8 + 8j, "z": b"eight"},
            {"x": 9.9, "y": 9 + 9j, "z": b"nine"},
        ]
    )

    kwargs = {
        "nan_string": "nan",
        "infinity_string": "inf",
        "minus_infinity_string": "-inf",
        "complex_record_fields": ("real", "imag"),
        "convert_bytes": lambda x: base64.b64encode(x).decode(),
    }

    expectation = '[{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"},{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"},{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="},{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="},{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="},{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"},{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="},{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="},{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}]'

    assert ak._v2.to_json(array, **kwargs) == expectation

    ak._v2.to_json(array, filename, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(array, file, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = '{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}'

    assert ak._v2.to_json(array[0], **kwargs) == expectation

    ak._v2.to_json(array[0], filename, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(array[0], file, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """[
    {
        "x": 1.1,
        "y": {
            "real": 1.0,
            "imag": 1.0
        },
        "z": "b25l"
    },
    {
        "x": 2.2,
        "y": {
            "real": 2.0,
            "imag": 2.0
        },
        "z": "dHdv"
    },
    {
        "x": 3.3,
        "y": {
            "real": 3.0,
            "imag": 3.0
        },
        "z": "dGhyZWU="
    },
    {
        "x": "nan",
        "y": {
            "real": "nan",
            "imag": 0.0
        },
        "z": "Zm91cg=="
    },
    {
        "x": "inf",
        "y": {
            "real": "inf",
            "imag": 5.0
        },
        "z": "Zml2ZQ=="
    },
    {
        "x": "-inf",
        "y": {
            "real": "nan",
            "imag": "-inf"
        },
        "z": "c2l4"
    },
    {
        "x": 7.7,
        "y": {
            "real": 7.0,
            "imag": 7.0
        },
        "z": "c2V2ZW4="
    },
    {
        "x": null,
        "y": {
            "real": 8.0,
            "imag": 8.0
        },
        "z": "ZWlnaHQ="
    },
    {
        "x": 9.9,
        "y": {
            "real": 9.0,
            "imag": 9.0
        },
        "z": "bmluZQ=="
    }
]"""

    assert (
        ak._v2.to_json(
            array, num_indent_spaces=4, num_readability_spaces=1, **kwargs
        ).replace(" \n", "\n")
        == expectation
    )

    ak._v2.to_json(
        array, filename, num_indent_spaces=4, num_readability_spaces=1, **kwargs
    )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(
            array, file, num_indent_spaces=4, num_readability_spaces=1, **kwargs
        )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    expectation = """{
    "x": 1.1,
    "y": {
        "real": 1.0,
        "imag": 1.0
    },
    "z": "b25l"
}"""

    assert (
        ak._v2.to_json(
            array[0], num_indent_spaces=4, num_readability_spaces=1, **kwargs
        ).replace(" \n", "\n")
        == expectation
    )

    ak._v2.to_json(
        array[0], filename, num_indent_spaces=4, num_readability_spaces=1, **kwargs
    )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(
            array[0], file, num_indent_spaces=4, num_readability_spaces=1, **kwargs
        )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"}
{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="}
{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="}
{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="}
{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"}
{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="}
{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="}
{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}
"""

    assert ak._v2.to_json(array, line_delimited=True, **kwargs) == expectation

    ak._v2.to_json(array, filename, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(array, file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
"""

    assert ak._v2.to_json(array[0], line_delimited=True, **kwargs) == expectation

    ak._v2.to_json(array[0], filename, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak._v2.to_json(array[0], file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"}
{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="}
{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="}
{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="}
{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"}
{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="}
{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="}
{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}
"""

    with open(filename, "w") as file:
        for x in array:
            ak._v2.to_json(x, file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation
