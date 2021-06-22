# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.types.type import Type


class ListType(Type):
    def __init__(self, content, parameters=None, typestr=None):
        if not isinstance(content, Type):
            raise TypeError(
                "{0} 'content' must be a Type subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
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
        self._content = content
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    def __str__(self):
        if self._typestr is not None:
            out = self._typestr

        elif self.parameter("__array__") == "string":
            return "string"

        elif self.parameter("__array__") == "bytestring":
            return "bytes"

        else:
            params = self._str_parameters()
            if params is None:
                out = "var * {0}".format(str(self._content))
            else:
                out = "[var * {0}, {1}]".format(str(self._content), params)

        return self._str_categorical_begin() + out + self._str_categorical_end()

    def __repr__(self):
        args = [repr(self._content)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))
