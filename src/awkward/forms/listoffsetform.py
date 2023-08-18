# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._parameters import type_parameters_equal
from awkward._typing import JSONMapping, JSONSerializable, final
from awkward._util import UNSET
from awkward.forms.form import Form


@final
class ListOffsetForm(Form):
    is_list = True

    def __init__(
        self,
        offsets: str,
        content: Form,
        *,
        parameters: JSONMapping | None = None,
        form_key: str | None = None,
    ):
        if not isinstance(offsets, str):
            raise TypeError(
                "{} 'offsets' must be of type str, not {}".format(
                    type(self).__name__, repr(offsets)
                )
            )

        self._offsets = offsets
        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    def copy(
        self,
        offsets=UNSET,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return ListOffsetForm(
            self._offsets if offsets is UNSET else offsets,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(cls, offsets, content, *, parameters=None, form_key=None):
        return cls(offsets, content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._offsets), repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "ListOffsetArray",
                "offsets": self._offsets,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.ListType(
            self._content.type,
            parameters=self._parameters,
        )

    def __eq__(self, other):
        if isinstance(other, ListOffsetForm):
            return (
                self._form_key == other._form_key
                and self._offsets == other._offsets
                and type_parameters_equal(self._parameters, other._parameters)
                and self._content == other._content
            )
        else:
            return False

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

        return self._content.purelist_parameters(*keys)

    @property
    def purelist_isregular(self):
        return False

    @property
    def purelist_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return 1
        else:
            return self._content.purelist_depth + 1

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (1, 1)
        else:
            mindepth, maxdepth = self._content.minmax_depth
            return (mindepth + 1, maxdepth + 1)

    @property
    def branch_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (False, 1)
        else:
            branch, depth = self._content.branch_depth
            return (branch, depth + 1)

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self):
        return self._content.is_tuple

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        if (
            self.parameter("__array__") not in ("string", "bytestring")
            and list_indicator is not None
        ):
            path = (*path, list_indicator)
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool):
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return ListOffsetForm(
                self._offsets,
                next_content,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _select_columns(self, match_specifier):
        return ListOffsetForm(
            self._offsets,
            self._content._select_columns(match_specifier),
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return ("string",)
        else:
            return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L419-L425
            has_identities, parameters, form_key, offsets, content = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(offsets, content, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(self, getkey):
        from awkward.operations.ak_from_buffers import _index_to_dtype

        yield (getkey(self, "offsets"), _index_to_dtype[self._offsets])
        yield from self._content._expected_from_buffers(getkey)
