# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable

import awkward as ak
from awkward._layout import wrap_layout
from awkward._namedaxis import _prettify_named_axes
from awkward._nplikes.numpy import Numpy, NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

if TYPE_CHECKING:
    from awkward.contents.content import Content


FormatterType: TypeAlias = "Callable[[Any], str]"


class FormatterOptions(TypedDict, total=False):
    bool: FormatterType
    int: FormatterType
    timedelta: FormatterType
    datetime: FormatterType
    float: FormatterType
    longfloat: FormatterType
    complexfloat: FormatterType
    longcomplexfloat: FormatterType
    numpystr: FormatterType
    object: FormatterType
    all: FormatterType
    int_kind: FormatterType
    float_kind: FormatterType
    complex_kind: FormatterType
    str_kind: FormatterType
    str: FormatterType
    bytes: FormatterType


np = NumpyMetadata.instance()
numpy = Numpy.instance()


def half(integer: int) -> int:
    return int(math.ceil(integer / 2))


def alternate(length: int):
    halfindex = half(length)
    forward = iter(range(halfindex))
    backward = iter(range(length - 1, halfindex - 1, -1))
    going_forward, going_backward = True, True
    while going_forward or going_backward:
        if going_forward:
            try:
                yield True, next(forward)
            except StopIteration:
                going_forward = False
        if going_backward:
            try:
                yield False, next(backward)
            except StopIteration:
                going_backward = False


is_identifier = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


# avoid recursion in which ak.Array.__getitem__ calls prettyprint
# to form an error string: private reimplementation of ak.Array.__getitem__


class PlaceholderValue:
    def __str__(self):
        return "XX"


class VirtualValue:
    def __str__(self):
        return "??"


def get_at(data: Content, index: int):
    if data._layout._is_getitem_at_placeholder():
        return PlaceholderValue()
    elif data._layout._is_getitem_at_virtual():
        return VirtualValue()
    out = data._layout._getitem_at(index)
    if isinstance(out, ak.contents.NumpyArray):
        array_param = out.parameter("__array__")
        if array_param == "byte":
            return ak._util.tobytes(out._raw(numpy))
        elif array_param == "char":
            return ak._util.tobytes(out._raw(numpy)).decode(errors="surrogateescape")
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return wrap_layout(out, data._behavior)
    else:
        return out


def get_field(data: Content, field: str):
    if isinstance(data._layout, ak.record.Record):
        if data._layout._array.content(field)._is_getitem_at_placeholder():
            return PlaceholderValue()
        elif data._layout._array.content(field)._is_getitem_at_virtual():
            return VirtualValue()
    out = data._layout._getitem_field(field)
    if isinstance(out, ak.contents.NumpyArray):
        array_param = out.parameter("__array__")
        if array_param == "byte":
            return ak._util.tobytes(out._raw(numpy))
        elif array_param == "char":
            return ak._util.tobytes(out._raw(numpy)).decode(errors="surrogateescape")
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return wrap_layout(out, data._behavior)
    else:
        return out


def custom_str(current: Any) -> str | None:
    if (
        issubclass(type(current), ak.highlevel.Record)
        and type(current).__str__ is not ak.highlevel.Record.__str__
    ) or (
        issubclass(type(current), ak.highlevel.Array)
        and type(current).__str__ is not ak.highlevel.Array.__str__
    ):
        return str(current)

    elif (
        issubclass(type(current), ak.highlevel.Record)
        and type(current).__repr__ is not ak.highlevel.Record.__repr__
    ) or (
        issubclass(type(current), ak.highlevel.Array)
        and type(current).__repr__ is not ak.highlevel.Array.__repr__
    ):
        return repr(current)

    else:
        return None


def valuestr_horiz(
    data: Any, limit_cols: int, formatter: Formatter
) -> tuple[int, list[str]]:
    if isinstance(data, (ak.highlevel.Array, ak.highlevel.Record)) and (
        not data.layout.backend.nplike.known_data
    ):
        if isinstance(data, ak.highlevel.Array):
            return 5, ["[...]"]

    original_limit_cols = limit_cols

    if isinstance(data, ak.highlevel.Array):
        front, back = ["["], ["]"]
        limit_cols -= 2

        if len(data) == 0:
            return 2, front + back

        elif len(data) == 1:
            cols_taken, strs = valuestr_horiz(get_at(data, 0), limit_cols, formatter)
            return 2 + cols_taken, front + strs + back

        else:
            limit_cols -= 5  # anticipate the ", ..."
            which = 0
            for forward, index in alternate(len(data)):
                current = get_at(data, index)
                if forward:
                    for_comma = 0 if which == 0 else 2
                    cols_taken, strs = valuestr_horiz(
                        current, limit_cols - for_comma, formatter
                    )

                    custom = custom_str(current)
                    if custom is not None:
                        strs = custom

                    if limit_cols - (for_comma + cols_taken) >= 0:
                        if which != 0:
                            front.append(", ")
                            limit_cols -= 2
                        front.extend(strs)
                        limit_cols -= cols_taken
                    else:
                        break
                else:
                    cols_taken, strs = valuestr_horiz(
                        current, limit_cols - 2, formatter
                    )

                    custom = custom_str(current)
                    if custom is not None:
                        strs = custom

                    if limit_cols - (2 + cols_taken) >= 0:
                        back[:0] = strs
                        back.insert(0, ", ")
                        limit_cols -= 2 + cols_taken
                    else:
                        break

                which += 1

            if which == 0:
                front.append("...")
                limit_cols -= 3
            elif which != len(data):
                front.append(", ...")
                limit_cols -= 5

            limit_cols += 5  # credit the ", ..."
            return original_limit_cols - limit_cols, front + back

    elif isinstance(data, ak.highlevel.Record):
        is_tuple = data.layout.is_tuple

        front = ["("] if is_tuple else ["{"]
        limit_cols -= 2  # both the opening and closing brackets
        limit_cols -= 5  # anticipate the ", ..."

        which = 0
        fields = data.fields
        for key in fields:
            for_comma = 0 if which == 0 else 2
            if is_tuple:
                key_str = ""
            else:
                if is_identifier.match(key) is None:
                    key_str = repr(key) + ": "
                    if key_str.startswith("u"):
                        key_str = key_str[1:]
                else:
                    key_str = key + ": "

            if limit_cols - (for_comma + len(key_str) + 3) >= 0:
                if which != 0:
                    front.append(", ")
                    limit_cols -= 2
                front.append(key_str)
                limit_cols -= len(key_str)
                which += 1

                target = limit_cols if len(fields) == 1 else half(limit_cols)
                cols_taken, strs = valuestr_horiz(
                    get_field(data, key), target, formatter
                )
                if limit_cols - cols_taken >= 0:
                    front.extend(strs)
                    limit_cols -= cols_taken
                else:
                    front.append("...")
                    limit_cols -= 3
                    break

            else:
                break

            which += 1

        if len(fields) != 0:
            if which == 0:
                front.append("...")
                limit_cols -= 3
            elif which != 2 * len(fields):
                front.append(", ...")
                limit_cols -= 5

        limit_cols += 5  # credit the ", ..."
        front.append(")" if is_tuple else "}")
        return original_limit_cols - limit_cols, front

    else:
        out = formatter(data)
        return len(out), [out]


class Formatter:
    def __init__(self, formatters: FormatterOptions | None = None, precision: int = 3):
        self._formatters: FormatterOptions = formatters or {}
        self._precision: int = precision
        self._cache: dict[type, FormatterType] = {}

    def __call__(self, obj: Any) -> str:
        try:
            impl = self._cache[type(obj)]
        except KeyError:
            impl = self._find_formatter_impl(type(obj))
            self._cache[type(obj)] = impl
        return impl(obj)

    def _format_complex(self, data: complex) -> str:
        return f"{data.real:.{self._precision}g}+{data.imag:.{self._precision}g}j"

    def _format_real(self, data: float) -> str:
        return f"{data:.{self._precision}g}"

    def _find_formatter_impl(self, cls: type) -> FormatterType:
        if issubclass(cls, np.bool_):
            try:
                return self._formatters["bool"]
            except KeyError:
                return str
        elif issubclass(cls, np.integer):
            try:
                return self._formatters["int"]
            except KeyError:
                return self._formatters.get("int_kind", str)
        elif issubclass(cls, (np.float64, np.float32)):
            try:
                return self._formatters["float"]
            except KeyError:
                return self._formatters.get("float_kind", self._format_real)
        elif hasattr(np, "float128") and issubclass(cls, np.float128):
            try:
                return self._formatters["longfloat"]
            except KeyError:
                return self._formatters.get("float_kind", self._format_real)
        elif issubclass(cls, (np.complex64, np.complex128)):
            try:
                return self._formatters["complexfloat"]
            except KeyError:
                return self._formatters.get("complex_kind", self._format_complex)
        elif hasattr(np, "complex256") and issubclass(cls, np.complex256):
            try:
                return self._formatters["longcomplexfloat"]
            except KeyError:
                return self._formatters.get("complex_kind", self._format_complex)
        elif issubclass(cls, np.datetime64):
            try:
                return self._formatters["datetime"]
            except KeyError:
                return str
        elif issubclass(cls, np.timedelta64):
            try:
                return self._formatters["timedelta"]
            except KeyError:
                return str
        elif issubclass(cls, str):
            try:
                return self._formatters["str"]
            except KeyError:
                return self._formatters.get("str_kind", repr)
        elif issubclass(cls, bytes):
            try:
                return self._formatters["bytes"]
            except KeyError:
                return self._formatters.get("str_kind", repr)
        else:
            return str


def valuestr(
    data: Any, limit_rows: int, limit_cols: int, formatter: Formatter | None = None
) -> str:
    if formatter is None:
        formatter = Formatter()

    if isinstance(data, (ak.highlevel.Array, ak.highlevel.Record)) and (
        not data.layout.backend.nplike.known_data
    ):
        if isinstance(data, ak.highlevel.Array):
            return "[...]"

    if limit_rows <= 1:
        _, strs = valuestr_horiz(data, limit_cols, formatter)
        return "".join(strs)

    elif isinstance(data, ak.highlevel.Array):
        front, back = [], []
        which = 0
        for forward, index in alternate(len(data)):
            _, strs = valuestr_horiz(get_at(data, index), limit_cols - 2, formatter)
            if forward:
                front.append("".join(strs))
            else:
                back.insert(0, "".join(strs))

            which += 1
            if which >= limit_rows:
                break

        if len(data) != 0 and which != len(data):
            back[0] = "..."

        out = front + back
        for i, val in enumerate(out):
            if i > 0:
                val = out[i] = " " + val
            else:
                val = out[i] = "[" + val
            if i < len(out) - 1:
                out[i] = val + ","
            else:
                out[i] = val + "]"

        return "\n".join(out)

    elif isinstance(data, ak.highlevel.Record):
        is_tuple = data.layout.is_tuple

        front = []

        which = 0
        fields = data.fields
        for key in fields:
            if is_tuple:
                key_str = ""
            else:
                if is_identifier.match(key) is None:
                    key_str = repr(key) + ": "
                    if key_str.startswith("u"):
                        key_str = key_str[1:]
                else:
                    key_str = key + ": "
            _, strs = valuestr_horiz(
                get_field(data, key), limit_cols - 2 - len(key_str), formatter
            )
            front.append(key_str + "".join(strs))

            which += 1
            if which >= limit_rows:
                break

        if len(fields) != 0 and which != len(fields):
            front[-1] = "..."

        out = front
        for i, val in enumerate(out):
            if i > 0:
                val = out[i] = " " + val
            elif data.is_tuple:
                val = out[i] = "(" + val
            else:
                val = out[i] = "{" + val
            if i < len(out) - 1:
                out[i] = val + ","
            elif data.is_tuple:
                out[i] = val + ")"
            else:
                out[i] = val + "}"
        return "\n".join(out)

    else:
        raise AssertionError(type(data))


def bytes_repr(nbytes: int) -> str:
    count, unit = (
        (f"{nbytes / 1e9:,.1f}", "GB")
        if nbytes > 1e9
        else (f"{nbytes / 1e6:,.1f}", "MB")
        if nbytes > 1e6
        else (f"{nbytes / 1e3:,.1f}", "kB")
        if nbytes > 1e3
        else (f"{nbytes:,}", "B")
    )

    return f"{count} {unit}"


def highlevel_array_show_rows(
    array,
    limit_rows=20,
    limit_cols=80,
    type=False,
    named_axis=False,
    nbytes=False,
    backend=False,
    *,
    formatter=None,
    precision=3,
) -> list[str]:
    rows = []
    formatter_impl = Formatter(formatter, precision=precision)

    array_line = valuestr(array, limit_rows, limit_cols, formatter=formatter_impl)
    rows.append(array_line)

    if type:
        typeio = io.StringIO()
        array.type.show(stream=typeio)
        type_line = "type: "
        type_line += typeio.getvalue().removesuffix("\n")
        rows.append(type_line)

    # other info
    if named_axis and array.named_axis:
        named_axis_line = "named axis: "
        named_axis_line += _prettify_named_axes(
            array.named_axis, delimiter=", ", maxlen=None
        )
        rows.append(named_axis_line)
    if nbytes:
        if array.nbytes is unknown_length:
            nbytes_line = "nbytes: unknown"
        else:
            nbytes_line = f"nbytes: {bytes_repr(array.nbytes)}"
        rows.append(nbytes_line)
    if backend:
        backend_line = f"backend: {array.layout.backend.name}"
        rows.append(backend_line)

    # make sure the type is always the second row, don't move it
    if type:
        assert rows[1].startswith("type: ")
    return rows
