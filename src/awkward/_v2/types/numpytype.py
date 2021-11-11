# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

from awkward._v2.types.type import Type
from awkward._v2.forms.form import _parameters_equal

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

# FIXME: this dict must become two functions (is_primitive and primitive_to_dtype)
# because NumPy takes ARBITRARY intervals, such as "15s".
_primitive_to_dtype = {
    "bool": np.dtype(np.bool_),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "complex64": np.dtype(np.complex64),
    "complex128": np.dtype(np.complex128),
    "datetime64": np.dtype(np.datetime64),
    "datetime64[Y]": np.dtype("datetime64[Y]"),
    "datetime64[M]": np.dtype("datetime64[M]"),
    "datetime64[W]": np.dtype("datetime64[W]"),
    "datetime64[D]": np.dtype("datetime64[D]"),
    "datetime64[h]": np.dtype("datetime64[h]"),
    "datetime64[m]": np.dtype("datetime64[m]"),
    "datetime64[s]": np.dtype("datetime64[s]"),
    "datetime64[ms]": np.dtype("datetime64[ms]"),
    "datetime64[us]": np.dtype("datetime64[us]"),
    "datetime64[ns]": np.dtype("datetime64[ns]"),
    "datetime64[ps]": np.dtype("datetime64[ps]"),
    "datetime64[fs]": np.dtype("datetime64[fs]"),
    "datetime64[as]": np.dtype("datetime64[as]"),
    "timedelta64": np.dtype(np.timedelta64),
    "timedelta64[Y]": np.dtype("timedelta64[Y]"),
    "timedelta64[M]": np.dtype("timedelta64[M]"),
    "timedelta64[W]": np.dtype("timedelta64[W]"),
    "timedelta64[D]": np.dtype("timedelta64[D]"),
    "timedelta64[h]": np.dtype("timedelta64[h]"),
    "timedelta64[m]": np.dtype("timedelta64[m]"),
    "timedelta64[s]": np.dtype("timedelta64[s]"),
    "timedelta64[ms]": np.dtype("timedelta64[ms]"),
    "timedelta64[us]": np.dtype("timedelta64[us]"),
    "timedelta64[ns]": np.dtype("timedelta64[ns]"),
    "timedelta64[ps]": np.dtype("timedelta64[ps]"),
    "timedelta64[fs]": np.dtype("timedelta64[fs]"),
    "timedelta64[as]": np.dtype("timedelta64[as]"),
}

if hasattr(np, "float16"):
    _primitive_to_dtype["float16"] = np.dtype(np.float16)
if hasattr(np, "float128"):
    _primitive_to_dtype["float128"] = np.dtype(np.float128)
if hasattr(np, "complex256"):
    _primitive_to_dtype["complex256"] = np.dtype(np.complex256)

_dtype_to_primitive = {}
for primitive, dtype in _primitive_to_dtype.items():
    _dtype_to_primitive[dtype] = primitive


class NumpyType(Type):
    def __init__(self, primitive, parameters=None, typestr=None):
        if primitive not in _primitive_to_dtype:
            raise TypeError(
                "{0} 'primitive' must be one of {1}, not {2}".format(
                    type(self).__name__,
                    ", ".join(repr(x) for x in _primitive_to_dtype),
                    repr(primitive),
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not ak._util.isstr(typestr):
            raise TypeError(
                "{0} 'typestr' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(typestr)
                )
            )
        self._primitive = primitive
        self._parameters = parameters
        self._typestr = typestr

    @property
    def primitive(self):
        return self._primitive

    _str_parameters_exclude = ("__categorical__", "__unit__")

    def __str__(self):
        if self._typestr is not None:
            out = self._typestr

        elif self.parameter("__array__") == "char":
            out = "char"

        elif self.parameter("__array__") == "byte":
            out = "byte"

        else:
            if self.parameter("__unit__") is not None:
                numpy_unit = str(np.dtype("M8[" + self._parameters["__unit__"] + "]"))
                bracket_index = numpy_unit.index("[")
                units = "unit=" + json.dumps(numpy_unit[bracket_index + 1 : -1])
            else:
                units = None

            params = self._str_parameters()

            if units is None and params is None:
                out = self._primitive
            else:
                if units is not None and params is not None:
                    units = units + ", "
                elif units is None:
                    units = ""
                elif params is None:
                    params = ""
                out = self._primitive + "[" + units + params + "]"

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = [repr(self._primitive)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, NumpyType):
            return (
                self._typestr == other._typestr
                and self._primitive == other._primitive
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
            )
        else:
            return False
