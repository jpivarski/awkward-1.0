# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import awkward as ak
from awkward._parameters import type_parameters_equal
from awkward._typing import Self, final
from awkward._util import UNSET
from awkward.forms.form import Form


@final
class ByteMaskedForm(Form):
    is_option = True

    def __init__(
        self,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(mask, str):
            raise TypeError(
                "{} 'mask' must be of type str, not {}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{} 'valid_when' must be bool, not {}".format(
                    type(self).__name__, repr(valid_when)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(parameters=parameters, form_key=form_key)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    def copy(
        self,
        mask=UNSET,
        content=UNSET,
        valid_when=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return ByteMaskedForm(
            self._mask if mask is UNSET else mask,
            self._content if content is UNSET else content,
            self._valid_when if valid_when is UNSET else valid_when,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        mask,
        content,
        valid_when,
        *,
        parameters=None,
        form_key=None,
    ):
        if content.is_union:
            return content._union_of_optionarrays("i64", parameters)
        elif content.is_indexed or content.is_option:
            return ak.forms.IndexedOptionForm.simplified(
                "i64",
                content,
                parameters=parameters,
            )
        else:
            return cls(
                mask, content, valid_when, parameters=parameters, form_key=form_key
            )

    @property
    def is_identity_like(self):
        return False

    def __repr__(self):
        args = [
            repr(self._mask),
            repr(self._content),
            repr(self._valid_when),
            *self._repr_args(),
        ]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "ByteMaskedArray",
                "mask": self._mask,
                "valid_when": self._valid_when,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.OptionType(
            self._content.type, parameters=self._parameters
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, ByteMaskedForm):
            return (
                self._form_key == other._form_key
                and self._mask == other._mask
                and self._valid_when == other._valid_when
                and type_parameters_equal(self._parameters, other._parameters)
                and self._content == other._content
            )
        else:
            return False

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return self._content.purelist_parameter(key)
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        return self._content.purelist_isregular

    @property
    def purelist_depth(self):
        return self._content.purelist_depth

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
            return ByteMaskedForm(
                self._mask,
                next_content,
                self._valid_when,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _select_columns(self, match_specifier):
        return ByteMaskedForm(
            self._mask,
            self._content._select_columns(match_specifier),
            self._valid_when,
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        return self._content._column_types()
