# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Collection

import awkward as ak
from awkward._backends.backend import Backend
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.virtual import VirtualArray
from awkward._typing import Callable, TypeAlias, TypeVar, cast
from awkward._util import UNSET, Sentinel

np = NumpyMetadata.instance()
numpy = Numpy.instance()

D = TypeVar("D")
T = TypeVar("T")

BackendLookup: TypeAlias = "Callable[[T], Backend]"
BackendLookupFactory: TypeAlias = "Callable[[type[T]], BackendLookup[T]]"


_type_to_backend_lookup: dict[type, BackendLookup] = {}
_backend_lookup_factories: list[BackendLookupFactory] = []
_name_to_backend_cls: dict[str, type[Backend]] = {}


def register_backend_lookup_factory(factory: BackendLookupFactory):
    _backend_lookup_factories.append(factory)


def register_backend(primary_nplike_cls: type[NumpyLike]):
    def wrapper(backend_cls: type[Backend]):
        def lookup(cls):
            return backend_cls.instance()

        def lookup_factory(cls):
            if primary_nplike_cls.is_own_array_type(cls):
                return lookup

        register_backend_lookup_factory(lookup_factory)

        _name_to_backend_cls[backend_cls.name] = backend_cls

        return backend_cls

    return wrapper


def common_backend(backends: Collection[Backend]) -> Backend:
    # Either we have one nplike, or one + typetracer
    if len(backends) == 1:
        return next(iter(backends))
    else:
        # We allow typetracers to mix with other nplikes, and take precedence
        for backend in backends:
            if not backend.nplike.known_data:
                return backend

        if len(backends) > 1:
            raise ValueError(
                "cannot operate on arrays with incompatible backends. Use #ak.to_backend to coerce the arrays "
                "to the same backend"
            )

        else:
            raise ValueError(
                "no backends were given in order to determine a common backend."
            )


def backend_of_obj(obj, default: D | Sentinel = UNSET) -> Backend | D:
    if isinstance(obj, VirtualArray):
        if isinstance(obj.nplike, (ak._nplikes.numpy.Numpy, ak._nplikes.cupy.Cupy)):
            cls = obj.nplike.ndarray
        else:
            raise ValueError(
                f"Only numpy and cupy nplikes are supported for VirtualArray. Received {type(obj.nplike)}"
            )
    else:
        cls = type(obj)
    try:
        lookup = _type_to_backend_lookup[cls]
        return lookup(obj)
    except KeyError:
        for factory in _backend_lookup_factories:
            maybe_lookup = factory(cls)
            if maybe_lookup is not None:
                break
        else:
            if default is UNSET:
                raise TypeError(f"cannot find backend for {cls.__name__}")
            else:
                return cast(D, default)
        _type_to_backend_lookup[cls] = maybe_lookup
        return maybe_lookup(obj)


def backend_of(
    *objects, default: D | Sentinel = UNSET, coerce_to_common: bool = True
) -> Backend | D:
    """
    Args:
        objects: objects for which to find a suitable backend
        default: value to return if no backend is found.
        coerce_to_common: try to coerce to a single backend if multiple backends found

    Return the most suitable backend for the given objects (e.g. arrays, layouts). If no
    suitable backend is found, return the `default` value, or raise a `ValueError` if
    no default is given.
    """
    unique_backends = frozenset(
        b for b in (backend_of_obj(o, default=None) for o in objects) if b is not None
    )

    if len(unique_backends) == 0:
        if default is UNSET:
            raise ValueError("could not find backend for", objects)
        else:
            return cast(D, default)
    elif len(unique_backends) == 1:
        return next(iter(unique_backends))
    elif coerce_to_common:
        return common_backend(unique_backends)
    else:
        raise ValueError(
            f"could not find singular backend for "
            f"{', '.join(type(t).__name__ for t in objects)} "
            f"and coercion is not permitted",
        )


def regularize_backend(backend: str | Backend) -> Backend:
    if isinstance(backend, Backend):
        return backend
    elif backend in _name_to_backend_cls:
        return _name_to_backend_cls[backend].instance()
    else:
        raise ValueError(f"No such backend {backend!r} exists.")
