import json

import lark

import awkward as ak

from awkward._v2.types.numpytype import NumpyType
from awkward._v2.types.unknowntype import UnknownType
from awkward._v2.types.regulartype import RegularType
from awkward._v2.types.listtype import ListType
from awkward._v2.types.optiontype import OptionType
from awkward._v2.types.recordtype import RecordType
from awkward._v2.types.uniontype import UnionType
from awkward._v2.types.arraytype import ArrayType


grammar = r"""
start: type

type: numpytype
    | unknowntype
    | regulartype
    | listtype
    | varlen_string
    | fixedlen_string
    | char
    | byte
    | option1
    | option2
    | list_parameters
    | categorical

numpytype: numpytype_name ("[" "parameters" "=" json_object "]")?

numpytype_name: DTYPE
              | DATETIME64
              | TIMEDELTA64

DTYPE: "bool"
     | "int8"
     | "uint8"
     | "int16"
     | "uint16"
     | "int32"
     | "uint32"
     | "int64"
     | "uint64"
     | "float32"
     | "float64"
     | "complex64"
     | "complex128"

DATETIME64:  /datetime64(\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\])?/
TIMEDELTA64: /timedelta64(\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\])?/

unknowntype: "unknown" ("[" "parameters" "=" json_object "]")?

regulartype: INT "*" type

listtype: "var" "*" type

varlen_string: "string" -> varlen_string
             | "bytes" -> varlen_bytestring

fixedlen_string: ("string" "[" INT "]") -> fixedlen_string
               | ("bytes" "[" INT "]") -> fixedlen_bytestring

char: "char"

byte: "byte"

option1: "?" type

option2: "option" "[" type ("," "parameters" "=" json_object)? "]"



list_parameters: "[" type "," "parameters" "=" json_object "]"

categorical: "categorical" "[" "type" "=" type "]"

json: ESCAPED_STRING -> string
    | SIGNED_NUMBER  -> number
    | "true"         -> true
    | "false"        -> false
    | "null"         -> null
    | json_array
    | json_object

json_array:  "[" [json ("," json)*] "]"
json_object: "{" [json_pair ("," json_pair)*] "}"
json_pair:   ESCAPED_STRING ":" json

%import common.INT
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
"""


class Transformer:
    @staticmethod
    def _parameters(args, i):
        if i < len(args):
            return args[i]
        else:
            return None

    def start(self, args):
        return args[0]

    def type(self, args):
        return args[0]

    def numpytype(self, args):
        return ak._v2.types.NumpyType(args[0], parameters=self._parameters(args, 1))

    def numpytype_name(self, args):
        return str(args[0])

    def unknowntype(self, args):
        return ak._v2.types.UnknownType(parameters=self._parameters(args, 0))

    def regulartype(self, args):
        return ak._v2.types.RegularType(args[1], int(args[0]))

    def listtype(self, args):
        return ak._v2.types.ListType(args[0])

    def varlen_string(self, args):
        return ak._v2.types.ListType(
            ak._v2.types.NumpyType("uint8", {"__array__": "char"}),
            {"__array__": "string"},
        )

    def varlen_bytestring(self, args):
        return ak._v2.types.ListType(
            ak._v2.types.NumpyType("uint8", {"__array__": "byte"}),
            {"__array__": "bytestring"},
        )

    def fixedlen_string(self, args):
        return ak._v2.types.RegularType(
            ak._v2.types.NumpyType("uint8", {"__array__": "char"}),
            int(args[0]),
            {"__array__": "string"},
        )

    def fixedlen_bytestring(self, args):
        return ak._v2.types.RegularType(
            ak._v2.types.NumpyType("uint8", {"__array__": "byte"}),
            int(args[0]),
            {"__array__": "bytestring"},
        )

    def char(self, args):
        return ak._v2.types.NumpyType("uint8", {"__array__": "char"})

    def byte(self, args):
        return ak._v2.types.NumpyType("uint8", {"__array__": "byte"})

    def option1(self, args):
        return ak._v2.types.OptionType(args[0])

    def option2(self, args):
        return ak._v2.types.OptionType(args[0], parameters=self._parameters(args, 1))

    def list_parameters(self, args):
        # modify recently created type object
        args[0].parameters.update(args[1])
        return args[0]

    def categorical(self, args):
        # modify recently created type object
        args[0].parameters["__categorical__"] = True
        return args[0]

    def json(self, args):
        return args[0]

    def json_object(self, args):
        return dict(args)

    def json_pair(self, args):
        return (json.loads(args[0]), args[1])

    def json_array(self, args):
        return list(args)

    def string(self, args):
        return json.loads(args[0])

    def number(self, args):
        try:
            return int(args[0])
        except ValueError:
            return float(args[0])

    def true(self, args):
        return True

    def false(self, args):
        return False

    def null(self, args):
        return None


parser = lark.Lark(grammar, parser="lalr", transformer=Transformer())


def test_numpytype_int32():
    t = NumpyType("int32")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64():
    t = NumpyType("datetime64")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_10s():
    t = NumpyType("datetime64[10s]")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_parameter():
    t = NumpyType("int32", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_parameter():
    t = NumpyType("datetime64", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_10s_parameter():
    t = NumpyType("datetime64[10s]", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_categorical():
    t = NumpyType("int32", {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_parameters_categorical():
    t = NumpyType("int32", {"__array__": "Something", "__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype():
    t = UnknownType()
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_parameter():
    t = UnknownType({"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_categorical():
    t = UnknownType({"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_categorical_parameter():
    t = UnknownType({"__array__": "Something", "__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype():
    t = RegularType(NumpyType("int32"), 5)
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_parameter():
    t = RegularType(NumpyType("int32"), 5, {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_categorical():
    t = RegularType(NumpyType("int32"), 5, {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_categorical_parameter():
    t = RegularType(
        NumpyType("int32"), 5, {"__categorical__": True, "__array__": "Something"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype():
    t = ListType(NumpyType("int32"))
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_parameter():
    t = ListType(NumpyType("int32"), {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_categorical():
    t = ListType(NumpyType("int32"), {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_categorical_parameter():
    t = ListType(
        NumpyType("int32"), {"__categorical__": True, "__array__": "Something"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_varlen_string():
    t = ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"})
    assert str(parser.parse(str(t))) == str(t)


def test_varlen_bytestring():
    t = ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "bytestring"})
    assert str(parser.parse(str(t))) == str(t)


def test_fixedlen_string():
    t = RegularType(
        NumpyType("uint8", {"__array__": "char"}), 5, {"__array__": "string"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_fixedlen_bytestring():
    t = RegularType(
        NumpyType("uint8", {"__array__": "byte"}), 5, {"__array__": "bytestring"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_char():
    t = NumpyType("uint8", {"__array__": "char"})
    assert str(parser.parse(str(t))) == str(t)


def test_byte():
    t = NumpyType("uint8", {"__array__": "byte"})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32():
    t = OptionType(NumpyType("int32"))
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_parameters():
    t = OptionType(NumpyType("int32"), {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_categorical():
    t = OptionType(NumpyType("int32"), {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_categorical_parameters():
    t = OptionType(
        NumpyType("int32"), {"__array__": "Something", "__categorical__": True}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_option_varlen_string():
    t = OptionType(
        ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"})
    )
    assert str(parser.parse(str(t))) == str(t)


def test_option_varlen_string_parameters():
    t = OptionType(
        ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"}),
        {"__array__": "Something"},
    )
    assert str(parser.parse(str(t))) == str(t)
