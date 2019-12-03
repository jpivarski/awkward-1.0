# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers
import json
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1.util
import awkward1.layout

def fromnumpy(array):
    def recurse(array):
        if len(array.shape) == 0:
            return awkward1.layout.NumpyArray(array.reshape(1))
        elif len(array.shape) == 1:
            return awkward1.layout.NumpyArray(array)
        else:
            return awkward1.layout.RegularArray(recurse(array.reshape((-1,) + array.shape[2:])), array.shape[1])
    return awkward1.Array(recurse(array))

def fromiter(iterable, initial=1024, resize=2.0):
    out = awkward1.layout.FillableArray(initial=initial, resize=resize)
    for x in iterable:
        out.fill(x)
    return awkward1.Array(out.snapshot())

def fromjson(source, initial=1024, resize=2.0, buffersize=65536):
    return awkward1.Array(awkward1.layout.fromjson(source, initial=initial, resize=resize, buffersize=buffersize))

def tolist(array):
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif isinstance(array, awkward1.Array):
        return tolist(array.layout)

    elif isinstance(array, awkward1.Record):
        return tolist(array.layout)

    elif isinstance(array, awkward1.layout.Record) and array.istuple:
        return tuple(tolist(x) for x in array.values())

    elif isinstance(array, awkward1.layout.Record):
        return {n: tolist(x) for n, x in array.items()}

    elif isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.layout.FillableArray):
        return [tolist(x) for x in array]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, awkward1.layout.Content):
        return [tolist(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

def tojson(array, *args, **kwargs):
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, awkward1.Array):
        return tojson(array.layout, *args, **kwargs)

    elif isinstance(array, awkward1.Record):
        return tojson(array.layout, *args, **kwargs)

    elif isinstance(array, awkward1.layout.Record):
        return array.tojson(*args, **kwargs)

    elif isinstance(array, numpy.ndarray):
        return awkward1.layout.NumpyArray(array).tojson(*args, **kwargs)

    elif isinstance(array, awkward1.layout.FillableArray):
        return array.snapshot().tojson(*args, **kwargs)

    elif isinstance(array, awkward1.layout.Content):
        return array.tojson(*args, **kwargs)

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")]
