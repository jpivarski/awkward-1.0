# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

pytest.importorskip("pyarrow")

string = ak.Array(
    [
        ["\u03b1\u03b2\u03b3", ""],
        [],
        ["\u2192\u03b4\u03b5\u2190", "\u03b6z z\u03b6", "abc"],
    ]
)
bytestring = ak.Array(
    [
        ["\u03b1\u03b2\u03b3".encode(), b""],
        [],
        ["\u2192\u03b4\u03b5\u2190".encode(), "\u03b6z z\u03b6".encode(), b"abc"],
    ]
)

string_padded = ak.Array(
    [
        ["      αβγ      ", "               "],
        [],
        ["     →δε←      ", "     ζz zζ     ", "      abc      "],
    ]
)
bytestring_padded = ak.Array(
    [
        [b"    \xce\xb1\xce\xb2\xce\xb3     ", b"               "],
        [],
        [
            b"  \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90   ",
            b"    \xce\xb6z z\xce\xb6    ",
            b"      abc      ",
        ],
    ]
)


def test_is_alnum():
    assert ak.str.is_alnum(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alnum(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]


def test_is_alpha():
    assert ak.str.is_alpha(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alpha(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]


def test_is_decimal():
    assert ak.str.is_decimal(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_decimal(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_digit():
    assert ak.str.is_digit(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_digit(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_lower():
    assert ak.str.is_lower(string).tolist() == [
        [True, False],
        [],
        [True, True, True],
    ]
    assert ak.str.is_lower(bytestring).tolist() == [
        [False, False],
        [],
        [False, True, True],
    ]


def test_is_numeric():
    assert ak.str.is_numeric(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_numeric(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_printable():
    assert ak.str.is_printable(string).tolist() == [
        [True, True],
        [],
        [True, True, True],
    ]
    assert ak.str.is_printable(bytestring).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]


def test_is_space():
    assert ak.str.is_space(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_space(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_upper():
    assert ak.str.is_upper(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_upper(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_title():
    assert ak.str.is_title(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_title(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_ascii():
    assert ak.str.is_ascii(string).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]
    assert ak.str.is_ascii(bytestring).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]


def test_capitalize():
    assert ak.str.capitalize(string).tolist() == [
        ["Αβγ", ""],
        [],
        ["→δε←", "Ζz zζ", "Abc"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.capitalize(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"Abc"],
    ]


def test_length():
    assert ak.str.length(string).tolist() == [
        [3, 0],
        [],
        [4, 5, 3],
    ]
    assert ak.str.length(bytestring).tolist() == [
        [6, 0],
        [],
        [10, 7, 3],
    ]


def test_lower():
    assert ak.str.lower(string).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.lower(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]


def test_swapcase():
    assert ak.str.swapcase(string).tolist() == [
        ["ΑΒΓ", ""],
        [],
        ["→ΔΕ←", "ΖZ ZΖ", "ABC"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.swapcase(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"ABC"],
    ]


def test_title():
    assert ak.str.title(string).tolist() == [
        ["Αβγ", ""],
        [],
        ["→Δε←", "Ζz Zζ", "Abc"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.title(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"Abc"],
    ]


def test_upper():
    assert ak.str.upper(string).tolist() == [
        ["ΑΒΓ", ""],
        [],
        ["→ΔΕ←", "ΖZ ZΖ", "ABC"],  # noqa: RUF001, RUF003 (we care about Ζ vs Z)
    ]
    assert ak.str.upper(bytestring).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζZ Zζ".encode(), b"ABC"],
    ]


def test_repeat():
    assert ak.str.repeat(string, 3).tolist() == [
        ["αβγαβγαβγ", ""],
        [],
        ["→δε←→δε←→δε←", "ζz zζζz zζζz zζ", "abcabcabc"],
    ]
    assert ak.str.repeat(bytestring, 3).tolist() == [
        ["αβγαβγαβγ".encode(), b""],
        [],
        ["→δε←→δε←→δε←".encode(), "ζz zζζz zζζz zζ".encode(), b"abcabcabc"],
    ]

    assert ak.str.repeat(string, [[3, 3], [], [2, 0, 1]]).tolist() == [
        ["αβγαβγαβγ", ""],
        [],
        ["→δε←→δε←", "", "abc"],
    ]
    assert ak.str.repeat(bytestring, [[3, 3], [], [2, 0, 1]]).tolist() == [
        ["αβγαβγαβγ".encode(), b""],
        [],
        ["→δε←→δε←".encode(), b"", b"abc"],
    ]


def test_replace_slice():
    assert ak.str.replace_slice(string[:, :1], 1, 2, "qj").tolist() == [
        ["αqjγ"],  # noqa: RUF001
        [],
        ["→qjε←"],
    ]
    assert ak.str.replace_slice(bytestring[:, :1], 1, 2, b"qj").tolist() == [
        [b"\xceqj\xce\xb2\xce\xb3"],
        [],
        [b"\xe2qj\x92\xce\xb4\xce\xb5\xe2\x86\x90"],
    ]


def test_reverse():
    assert ak.str.reverse(string).tolist() == [
        ["αβγ"[::-1], ""],
        [],
        ["→δε←"[::-1], "ζz zζ"[::-1], "abc"[::-1]],
    ]
    assert ak.str.reverse(bytestring).tolist() == [
        ["αβγ".encode()[::-1], b""],
        [],
        ["→δε←".encode()[::-1], "ζz zζ".encode()[::-1], b"abc"[::-1]],
    ]


def test_replace_substring():
    assert ak.str.replace_substring(string, "βγ", "HELLO").tolist() == [
        ["αHELLO", ""],  # noqa: RUF001
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring(bytestring, "βγ".encode(), b"HELLO").tolist() == [
        ["αHELLO".encode(), b""],  # noqa: RUF001
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]

    assert ak.str.replace_substring(
        string, "βγ", "HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring(
        bytestring, "βγ".encode(), b"HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]


def test_replace_substring_regex():
    assert ak.str.replace_substring_regex(string, "βγ", "HELLO").tolist() == [
        ["αHELLO", ""],  # noqa: RUF001
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring_regex(
        bytestring, "βγ".encode(), b"HELLO"
    ).tolist() == [
        ["αHELLO".encode(), b""],  # noqa: RUF001
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]

    assert ak.str.replace_substring_regex(
        string, "βγ", "HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.replace_substring_regex(
        bytestring, "βγ".encode(), b"HELLO", max_replacements=0
    ).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]


def test_center():
    assert ak.str.center(string, 15, " ").tolist() == [
        ["      αβγ      ", "               "],
        [],
        ["     →δε←      ", "     ζz zζ     ", "      abc      "],
    ]

    print(ak.str.center(bytestring, 15, " ").tolist())

    assert ak.str.center(bytestring, 15, b" ").tolist() == [
        [b"    \xce\xb1\xce\xb2\xce\xb3     ", b"               "],
        [],
        [
            b"  \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90   ",
            b"    \xce\xb6z z\xce\xb6    ",
            b"      abc      ",
        ],
    ]


def test_lpad():
    assert ak.str.lpad(string, 15, " ").tolist() == [
        ["            αβγ", "               "],
        [],
        ["           →δε←", "          ζz zζ", "            abc"],
    ]

    print(ak.str.lpad(bytestring, 15, " ").tolist())

    assert ak.str.lpad(bytestring, 15, b" ").tolist() == [
        [b"         \xce\xb1\xce\xb2\xce\xb3", b"               "],
        [],
        [
            b"     \xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90",
            b"        \xce\xb6z z\xce\xb6",
            b"            abc",
        ],
    ]


def test_rpad():
    assert ak.str.rpad(string, 15, " ").tolist() == [
        ["αβγ            ", "               "],
        [],
        ["→δε←           ", "ζz zζ          ", "abc            "],
    ]

    print(ak.str.rpad(bytestring, 15, " ").tolist())

    assert ak.str.rpad(bytestring, 15, b" ").tolist() == [
        [b"\xce\xb1\xce\xb2\xce\xb3         ", b"               "],
        [],
        [
            b"\xe2\x86\x92\xce\xb4\xce\xb5\xe2\x86\x90     ",
            b"\xce\xb6z z\xce\xb6        ",
            b"abc            ",
        ],
    ]


def test_ltrim():
    assert ak.str.ltrim(string_padded, " ").tolist() == [
        ["αβγ      ", ""],
        [],
        ["→δε←      ", "ζz zζ     ", "abc      "],
    ]
    assert ak.str.ltrim(bytestring_padded, b" ").tolist() == [
        ["αβγ     ".encode(), b""],
        [],
        ["→δε←   ".encode(), "ζz zζ    ".encode(), b"abc      "],
    ]


def test_ltrim_whitespace():
    assert ak.str.ltrim_whitespace(string_padded).tolist() == [
        ["αβγ      ", ""],
        [],
        ["→δε←      ", "ζz zζ     ", "abc      "],
    ]
    assert ak.str.ltrim_whitespace(bytestring_padded).tolist() == [
        ["αβγ     ".encode(), b""],
        [],
        ["→δε←   ".encode(), "ζz zζ    ".encode(), b"abc      "],
    ]


def test_rtrim():
    assert ak.str.rtrim(string_padded, " ").tolist() == [
        ["      αβγ", ""],
        [],
        ["     →δε←", "     ζz zζ", "      abc"],
    ]
    assert ak.str.rtrim(bytestring_padded, b" ").tolist() == [
        ["    αβγ".encode(), b""],
        [],
        ["  →δε←".encode(), "    ζz zζ".encode(), b"      abc"],
    ]


def test_rtrim_whitespace():
    assert ak.str.rtrim_whitespace(string_padded).tolist() == [
        ["      αβγ", ""],
        [],
        ["     →δε←", "     ζz zζ", "      abc"],
    ]
    assert ak.str.rtrim_whitespace(bytestring_padded).tolist() == [
        ["    αβγ".encode(), b""],
        [],
        ["  →δε←".encode(), "    ζz zζ".encode(), b"      abc"],
    ]


def test_trim():
    assert ak.str.trim(string_padded, " ").tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.trim(bytestring_padded, b" ").tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]


def test_trim_whitespace():
    assert ak.str.trim_whitespace(string_padded).tolist() == [
        ["αβγ", ""],
        [],
        ["→δε←", "ζz zζ", "abc"],
    ]
    assert ak.str.trim_whitespace(bytestring_padded).tolist() == [
        ["αβγ".encode(), b""],
        [],
        ["→δε←".encode(), "ζz zζ".encode(), b"abc"],
    ]


def test_slice():
    assert ak.str.slice(string, 1, 3).tolist() == [
        ["αβγ"[1:3], ""[1:3]],
        [],
        ["→δε←"[1:3], "ζz zζ"[1:3], "abc"[1:3]],
    ]
    assert ak.str.slice(bytestring, 1, 3).tolist() == [
        ["αβγ".encode()[1:3], b""[1:3]],
        [],
        ["→δε←".encode()[1:3], "ζz zζ".encode()[1:3], b"abc"[1:3]],
    ]

    # ArrowInvalid: Negative buffer resize: -40 (looks like an Arrow bug)
    # assert ak.str.slice(string, 1).tolist() == [
    #     ["αβγ"[1:], ""[1:]],
    #     [],
    #     ["→δε←"[1:], "ζz zζ"[1:], "abc"[1:]],
    # ]
    assert ak.str.slice(bytestring, 1).tolist() == [
        ["αβγ".encode()[1:], b""[1:]],
        [],
        ["→δε←".encode()[1:], "ζz zζ".encode()[1:], b"abc"[1:]],
    ]
