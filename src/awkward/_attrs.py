# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from awkward._typing import Any, JSONMapping


def attrs_of_obj(obj, attrs: Mapping | None = None) -> Mapping | None:
    from awkward.highlevel import Array, ArrayBuilder, Record

    if attrs is not None:
        return attrs
    elif isinstance(obj, (Array, Record, ArrayBuilder)):
        return obj._attrs
    else:
        return None


def attrs_of(*arrays, attrs: Mapping | None = None) -> Mapping:
    # An explicit 'attrs' always wins.
    if attrs is not None:
        return attrs

    copied = False
    for x in reversed(arrays):
        x_attrs = attrs_of_obj(x)
        if x_attrs is None:
            continue
        if attrs is None:
            attrs = x_attrs
        elif attrs is x_attrs:
            pass
        elif not copied:
            attrs = dict(attrs)
            attrs.update(x_attrs)
            copied = True
        else:
            attrs.update(x_attrs)
    return attrs


def without_transient_attrs(attrs: dict[str, Any]) -> JSONMapping:
    return {k: v for k, v in attrs.items() if not k.startswith("@")}


class Attrs(Mapping):
    def __init__(self, ref, data: Mapping[str, Any]):
        self._ref = ref
        self._data = _freeze_attrs(data)

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._ref._attrs = _unfreeze_attrs(self._data) | {key: value}

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(_unfreeze_attrs(self._data))


def _freeze_attrs(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(attrs)


def _unfreeze_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return dict(attrs)
