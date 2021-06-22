# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.types.type import Type


class UnknownType(Type):
    def __init__(self, parameters=None, typestr=None):
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise TypeError(
                "{0} 'typestr' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(typestr)
                )
            )
        self._parameters = parameters
        self._typestr = typestr

    def __str__(self):
        if self._typestr is not None:
            out = self._typestr

        else:
            params = self._str_parameters()
            if params is None:
                out = "unknown"
            else:
                out = "unknown[" + params + "]"

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))
