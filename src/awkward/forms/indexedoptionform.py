# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import awkward as ak
from awkward._parameters import parameters_union, type_parameters_equal
from awkward._typing import JSONSerializable, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form


@final
class IndexedOptionForm(Form):
    is_option = True
    is_indexed = True

    def __init__(
        self,
        index,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(index, str):
            raise TypeError(
                f"{type(self).__name__} 'index' must be of type str, not {index!r}"
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._index = index
        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    def copy(
        self,
        index=UNSET,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return IndexedOptionForm(
            self._index if index is UNSET else index,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        index,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        is_cat = parameters is not None and parameters.get("__array__") == "categorical"

        if content.is_union and not is_cat:
            return content._union_of_optionarrays(index, parameters)

        elif content.is_indexed or content.is_option:
            return ak.forms.IndexedOptionForm.simplified(
                "i64",
                content.content,
                parameters=parameters_union(content._parameters, parameters),
            )

        else:
            return cls(index, content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._index), repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "IndexedOptionArray",
                "index": self._index,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        if self.parameter("__array__") == "categorical":
            parameters = dict(self._parameters)
            del parameters["__array__"]
            parameters["__categorical__"] = True
        else:
            parameters = self._parameters

        return ak.types.OptionType(
            self._content.type,
            parameters=parameters,
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, IndexedOptionForm):
            return (
                self._form_key == other._form_key
                and self._index == other._index
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
        return self._content.purelist_isregular

    @property
    def purelist_depth(self):
        return self._content.purelist_depth

    @property
    def is_identity_like(self):
        return self._content.is_identity_like

    @property
    def minmax_depth(self):
        return self._content.minmax_depth

    @property
    def branch_depth(self):
        return self._content.branch_depth

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self):
        return self._content.is_tuple

    @property
    def dimension_optiontype(self):
        return True

    def _columns(self, path, output, list_indicator):
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self | None:
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return IndexedOptionForm(
                self._index,
                next_content,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _select_columns(self, match_specifier):
        return IndexedOptionForm(
            self._index,
            self._content._select_columns(match_specifier),
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L324-L330
            has_identities, parameters, form_key, index, content = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(index, content, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(self, getkey):
        from awkward.operations.ak_from_buffers import _index_to_dtype

        yield (getkey(self, "index"), _index_to_dtype[self._index])
        yield from self._content._expected_from_buffers(getkey)
