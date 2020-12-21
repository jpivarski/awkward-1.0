# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def single(obj):
    if isinstance(obj, ak.layout.Content):
        return IrregularlyPartitionedArray([obj])
    else:
        return obj


def first(obj):
    if isinstance(obj, PartitionedArray):
        return obj.partition(0)
    else:
        return obj


def every(obj):
    if isinstance(obj, PartitionedArray):
        return obj.partitions
    else:
        return [obj]


def partition_as(sample, arrays):
    stops = sample.stops
    if isinstance(arrays, dict):
        out = {}
        for n, x in arrays.items():
            if isinstance(x, PartitionedArray):
                out[n] = x.repartition(stops)
            elif isinstance(x, ak.layout.Content) and (
                x.parameter("__array__") == "string"
                or x.parameter("__array__") == "bytestring"
            ):
                out[n] = x
            elif isinstance(x, ak.layout.Content):
                out[n] = IrregularlyPartitionedArray.toPartitioned(x, stops)
            else:
                out[n] = x
        return out
    else:
        out = []
        for x in arrays:
            if isinstance(x, PartitionedArray):
                out.append(x.repartition(stops))
            elif isinstance(x, ak.layout.Content) and (
                x.parameter("__array__") == "string"
                or x.parameter("__array__") == "bytestring"
            ):
                out.append(x)
            elif isinstance(x, ak.layout.Content):
                out.append(IrregularlyPartitionedArray.toPartitioned(x, stops))
            else:
                out.append(x)
        return out


def iterate(numpartitions, arrays):
    if isinstance(arrays, dict):
        for partitionid in range(numpartitions):
            out = {}
            for n, x in arrays.items():
                if isinstance(x, PartitionedArray):
                    out[n] = x.partition(partitionid)
                else:
                    out[n] = x
            yield out
    else:
        for partitionid in range(numpartitions):
            out = []
            for x in arrays:
                if isinstance(x, PartitionedArray):
                    out.append(x.partition(partitionid))
                else:
                    out.append(x)
            yield out


def apply(function, array):
    return IrregularlyPartitionedArray([function(x) for x in array.partitions])


class PartitionedArray(object):
    @classmethod
    def from_ext(cls, obj):
        if isinstance(obj, ak._ext.IrregularlyPartitionedArray):
            subcls = IrregularlyPartitionedArray
            out = subcls.__new__(subcls)
            out._ext = obj
            return out
        else:
            return obj

    def __repr__(self):
        return repr(self._ext)

    @property
    def partitions(self):
        return self._ext.partitions

    @property
    def numpartitions(self):
        return self._ext.numpartitions

    def partition(self, partitionid):
        return self._ext.partition(partitionid)

    @property
    def stops(self):
        total_length = 0
        out = []
        for x in self.partitions:
            total_length += len(x)
            out.append(total_length)
        return out

    @property
    def lengths(self):
        return [len(x) for x in self.partitions]

    def start(self, partitionid):
        return self._ext.start(partitionid)

    def stop(self, partitionid):
        return self._ext.stop(partitionid)

    def partitionid_index_at(self, at):
        return self._ext.partitionid_index_at(at)

    def type(self, typestrs):
        out = None
        for x in self.partitions:
            if out is None:
                out = x.type(typestrs)
            elif out != x.type(typestrs):
                raise ValueError(
                    "inconsistent types in PartitionedArray"
                    + ak._util.exception_suffix(__file__)
                )
        return out

    @property
    def parameters(self):
        return first(self).parameters

    def parameter(self, *args, **kwargs):
        return first(self).parameter(*args, **kwargs)

    def purelist_parameter(self, *args, **kwargs):
        return first(self).purelist_parameter(*args, **kwargs)

    @property
    def kernels(self):
        out = None
        for x in self.partitions:
            if out is None:
                out = x.kernels
            elif out != x.kernels:
                return "mixed"
        if out is None:
            return "cpu"
        else:
            return out

    def tojson(self, *args, **kwargs):
        return self._ext.tojson(*args, **kwargs)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.partitions)

    def deep_copy(self, *args, **kwargs):
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out._partitions = [x.deep_copy(*args, **kwargs) for x in out.partitions]
        return out

    @property
    def numfields(self):
        return first(self).numfields

    def fieldindex(self, *args, **kwargs):
        return first(self).fieldindex(*args, **kwargs)

    def key(self, *args, **kwargs):
        return first(self).key(*args, **kwargs)

    def haskey(self, *args, **kwargs):
        return first(self).haskey(*args, **kwargs)

    def keys(self, *args, **kwargs):
        return first(self).keys(*args, **kwargs)

    @property
    def purelist_isregular(self):
        return first(self).purelist_isregular

    @property
    def purelist_depth(self):
        return first(self).purelist_depth

    @property
    def branch_depth(self):
        return first(self).branch_depth

    @property
    def minmax_depth(self):
        return first(self).minmax_depth

    def getitem_nothing(self, *args, **kwargs):
        return first(self).getitem_nothing(*args, **kwargs)

    def copy_to(self, ptr_lib):
        return self.from_ext(self._ext.copy_to(ptr_lib))

    def validityerror(self, *args, **kwargs):
        t = None
        for x in self.partitions:
            if t is None:
                t = x.type({})
            elif t != x.type({}):
                return "inconsistent types in PartitionedArray"
            out = x.validityerror(*args, **kwargs)
            if out is not None:
                return out
        return None

    def num(self, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            return sum(x.num(axis) for x in self.partitions)
        else:
            return self.replace_partitions([x.num(axis) for x in self.partitions])

    def flatten(self, *args, **kwargs):
        return apply(lambda x: x.flatten(*args, **kwargs), self)

    def rpad(self, length, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            return self.toContent().rpad(length, axis)
        else:
            return self.replace_partitions(
                [x.rpad(length, axis) for x in self.partitions]
            )

    def rpad_and_clip(self, length, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            return self.toContent().rpad_and_clip(length, axis)
        else:
            return self.replace_partitions(
                [x.rpad_and_clip(length, axis) for x in self.partitions]
            )

    def reduce(self, name, axis, mask, keepdims):
        branch, depth = first(self).branch_depth
        negaxis = -axis
        if not branch and negaxis <= 0:
            negaxis += depth
        if not branch and negaxis == depth:
            return getattr(self.toContent(), name)(axis, mask, keepdims)
        else:
            return self.replace_partitions(
                [getattr(x, name)(axis, mask, keepdims) for x in self.partitions]
            )

    def count(self, axis, mask, keepdims):
        return self.reduce("count", axis, mask, keepdims)

    def count_nonzero(self, axis, mask, keepdims):
        return self.reduce("count_nonzero", axis, mask, keepdims)

    def sum(self, axis, mask, keepdims):
        return self.reduce("sum", axis, mask, keepdims)

    def prod(self, axis, mask, keepdims):
        return self.reduce("prod", axis, mask, keepdims)

    def any(self, axis, mask, keepdims):
        return self.reduce("any", axis, mask, keepdims)

    def all(self, axis, mask, keepdims):
        return self.reduce("all", axis, mask, keepdims)

    def min(self, axis, mask, keepdims):
        return self.reduce("min", axis, mask, keepdims)

    def max(self, axis, mask, keepdims):
        return self.reduce("max", axis, mask, keepdims)

    def argmin(self, axis, mask, keepdims):
        return self.reduce("argmin", axis, mask, keepdims)

    def argmax(self, axis, mask, keepdims):
        return self.reduce("argmax", axis, mask, keepdims)

    def axis_wrap_if_negative(self, axis):
        out = None
        for partition in self.partitions:
            if out is None:
                out = partition.axis_wrap_if_negative(axis)
            elif out != partition.axis_wrap_if_negative(axis):
                raise ValueError(
                    "partitions have inconsistent depths"
                    + ak._util.exception_suffix(__file__)
                )
        return out

    def localindex(self, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            start = 0
            output = []
            for x in self.partitions:
                output.append(
                    ak.layout.NumpyArray(
                        ak.nplike.of(x).arange(start, start + len(x), dtype=np.int64)
                    )
                )
                start += len(x)
            return self.replace_partitions(output)

        else:
            return self.replace_partitions(
                [x.localindex(axis) for x in self.partitions]
            )

    def combinations(self, n, replacement, keys, parameters, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            return self.toContent().combinations(n, replacement, keys, parameters, axis)
        else:
            return self.replace_partitions(
                [
                    x.combinations(n, replacement, keys, parameters, axis)
                    for x in self.partitions
                ]
            )

    def __len__(self):
        return len(self._ext)

    def __iter__(self):
        for partition in self.partitions:
            for x in partition:
                yield x

    def __array__(self):
        tocat = []
        for x in self.partitions:
            y = ak.operations.convert.to_numpy(x)
            if len(y) > 0:
                tocat.append(y)

        if len(tocat) > 0:
            if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
                return numpy.ma.concatenate(tocat)
            else:
                return ak.nplike.of(tocat).concatenate(tocat)
        else:
            return y

    def toContent(self):
        contents = self._ext.partitions
        out = contents[0]
        for x in contents[1:]:
            if not out.mergeable(x, mergebool=False):
                out = out.merge_as_union(x)
            else:
                out = out.merge(x)
            if isinstance(out, ak._util.uniontypes):
                out = out.simplify(mergebool=False)
        return out

    def repartition(self, *args, **kwargs):
        return PartitionedArray.from_ext(self._ext.repartition(*args, **kwargs))

    def __getitem__(self, where):
        if not isinstance(where, (bool, np.bool_)) and isinstance(
            where, (numbers.Integral, np.integer)
        ):
            return PartitionedArray.from_ext(self._ext.getitem_at(where))

        elif isinstance(where, slice):
            return PartitionedArray.from_ext(
                self._ext.getitem_range(where.start, where.stop, where.step)
            )

        elif isinstance(where, str) or (
            ak._util.py27 and isinstance(where, ak._util.unicode)
        ):
            return self.replace_partitions([x[where] for x in self.partitions])

        elif isinstance(where, tuple) and len(where) == 0:
            return self

        elif (
            isinstance(where, Iterable)
            and len(where) > 0
            and all(
                (
                    isinstance(x, str)
                    or (ak._util.py27 and isinstance(x, ak._util.unicode))
                )
                for x in where
            )
        ):
            return self.replace_partitions([x[where] for x in self.partitions])

        else:
            if not isinstance(where, tuple):
                where = (where,)
            head, tail = where[0], where[1:]

            if not isinstance(head, (bool, np.bool_)) and isinstance(
                head, (numbers.Integral, np.integer)
            ):
                if head < 0:
                    head += len(self)
                if not 0 <= head < len(self):
                    raise ValueError(
                        "{0} index out of range".format(type(self).__name__)
                        + ak._util.exception_suffix(__file__)
                    )
                partitionid, index = self._ext.partitionid_index_at(head)
                return self.partition(partitionid)[(index,) + tail]

            elif isinstance(head, slice):
                partitions = self._ext.getitem_range(
                    head.start, head.stop, head.step
                ).partitions
                return IrregularlyPartitionedArray(
                    [x[(slice(None),) + tail] for x in partitions]
                )

            elif head is Ellipsis:
                return IrregularlyPartitionedArray(
                    [x[(head,) + tail] for x in self.partitions]
                )

            elif isinstance(head, str) or (
                ak._util.py27 and isinstance(head, ak._util.unicode)
            ):
                y = IrregularlyPartitionedArray([x[head] for x in self.partitions])
                if len(tail) == 0:
                    return y
                else:
                    return y[tail]

            elif isinstance(head, Iterable) and all(
                (
                    isinstance(x, str)
                    or (ak._util.py27 and isinstance(x, ak._util.unicode))
                )
                for x in head
            ):
                y = IrregularlyPartitionedArray(
                    [x[list(head)] for x in self.partitions]
                )
                if len(tail) == 0:
                    return y
                else:
                    return y[tail]

            elif head is np.newaxis:
                return self.toContent()[(head,) + tail]

            else:
                layout = ak.operations.convert.to_layout(
                    head,
                    allow_record=False,
                    allow_other=False,
                    numpytype=(np.integer, np.bool_),
                )

                t = ak.operations.describe.type(layout)
                int_types = [
                    ak.types.PrimitiveType("int8"),
                    ak.types.PrimitiveType("uint8"),
                    ak.types.PrimitiveType("int16"),
                    ak.types.PrimitiveType("uint16"),
                    ak.types.PrimitiveType("int32"),
                    ak.types.PrimitiveType("uint32"),
                    ak.types.PrimitiveType("int64"),
                    ak.types.PrimitiveType("uint64"),
                ]
                opt_types = [ak.types.OptionType(p) for p in int_types]

                if (
                    isinstance(t, ak.types.OptionType)
                    or t in int_types + opt_types
                    or (
                        isinstance(t, ak.types.UnionType)
                        and all(ti in int_types for ti in t.types)
                    )
                    or (
                        isinstance(t, ak.types.OptionType)
                        and isinstance(t.type, ak.types.UnionType)
                        and all(ti in int_types for ti in t.type.types)
                    )
                ):

                    if isinstance(layout, PartitionedArray):
                        layout = layout.toContent()
                    return self.toContent()[(layout,) + tail]

                else:
                    stops = self.stops
                    if isinstance(layout, PartitionedArray):
                        layout = layout.repartition(stops)
                    else:
                        layout = IrregularlyPartitionedArray.toPartitioned(
                            layout, stops
                        )
                    inparts = self.partitions
                    headparts = layout.partitions
                    outparts = []
                    for i in range(len(inparts)):
                        outparts.append(inparts[i][(headparts[i],) + tail])
                    return IrregularlyPartitionedArray(outparts, stops)


class IrregularlyPartitionedArray(PartitionedArray):
    @classmethod
    def toPartitioned(cls, layout, stops):
        partitions = []
        start = 0
        for stop in stops:
            partitions.append(layout[start:stop])
            start = stop
        return IrregularlyPartitionedArray(partitions, stops)

    def __init__(self, partitions, stops=None):
        if stops is None:
            self._ext = ak._ext.IrregularlyPartitionedArray(partitions)
        else:
            self._ext = ak._ext.IrregularlyPartitionedArray(partitions, stops)

    @property
    def stops(self):
        return self._ext.stops

    def validityerror(self):
        total_length = 0
        for x, stop in zip(self.partitions, self.stops):
            total_length += len(x)
            if total_length != stop:
                return (
                    "IrregularlyPartitionedArray stops do not match "
                    "partition lengths"
                )
            out = x.validityerror()
            if out is not None:
                return out
        return None

    def replace_partitions(self, partitions):
        return IrregularlyPartitionedArray(partitions, self.stops)
