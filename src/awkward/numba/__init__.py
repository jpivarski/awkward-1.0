# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import numpy
from packaging.version import parse as parse_version

import awkward as ak
import awkward.numba.layoutbuilder

_has_checked_version = False
_is_registered = False


def register_and_check():
    global _has_checked_version

    try:
        import numba
    except ImportError as err:
        raise ImportError(
            """install the 'numba' package with:

pip install numba --upgrade

or

conda install numba"""
        ) from err

    if not _has_checked_version:
        if parse_version(numba.__version__) < parse_version("0.50"):
            raise ImportError(
                "Awkward Array can only work with numba 0.50 or later "
                f"(you have version {numba.__version__})"
            )
        _has_checked_version = True

    _register()


class GrowableBuffer:
    def __init__(self, dtype, *, initial=1024, resize=8.0):
        register_and_check()

        import numba

        # all mutable data are in arrays that can be in-place shared with Numba
        self._panels = numba.typed.List([numpy.empty((initial,), dtype=dtype)])
        self._length_pos = numpy.zeros((2,), dtype=numpy.int64)
        self._resize = resize

    @classmethod
    def _from_data(cls, panels, length_pos, resize):
        out = cls.__new__(cls)
        out._panels = panels
        out._length_pos = length_pos
        out._resize = resize
        return out

    @property
    def dtype(self):
        return self._panels[0].dtype

    def __repr__(self):
        return f"<GrowableBuffer of {self.dtype!r} with {self._length} items>"

    @property
    def _length(self):
        return self._length_pos[0]

    @_length.setter
    def _length(self, value):
        self._length_pos[0] = value

    def _length_inc(self, value):
        self._length_pos[0] += value

    @property
    def _pos(self):
        return self._length_pos[1]

    @_pos.setter
    def _pos(self, value):
        self._length_pos[1] = value

    def _pos_inc(self, value):
        self._length_pos[1] += value

    def __len__(self):
        return self._length

    def last(self):
        return self._panels[-1][self._pos - 1]

    def append(self, datum):
        if self._pos == len(self._panels[-1]):
            self._add_panel()

        self._panels[-1][self._pos] = datum
        self._pos_inc(1)
        self._length_inc(1)

    def extend(self, data):
        panel_index = len(self._panels) - 1
        pos = self._pos

        available = len(self._panels[-1]) - pos
        remaining = len(data)

        if remaining > available:
            panel_length = int(math.ceil(len(self._panels[0]) * self._resize))

            self._panels.append(
                numpy.empty((max(remaining, panel_length),), dtype=self.dtype)
            )
            self._pos = 0
            available += len(self._panels[-1])

        while remaining > 0:
            panel = self._panels[panel_index]
            available_in_panel = len(panel) - pos
            to_write = min(remaining, available_in_panel)

            start = len(data) - remaining
            panel[pos : pos + to_write] = data[start : start + to_write]

            if panel_index == len(self._panels) - 1:
                self._pos_inc(to_write)
            remaining -= to_write
            pos = 0
            panel_index += 1

        self._length_inc(len(data))

    def _add_panel(self):
        panel_length = int(math.ceil(len(self._panels[0]) * self._resize))

        self._panels.append(numpy.empty((panel_length,), dtype=self.dtype))
        self._pos = 0

    def snapshot(self):
        out = numpy.empty((self._length,), dtype=self.dtype)

        start = 0
        stop = 0
        for panel in self._panels[:-1]:  # full panels, not including the last
            stop += len(panel)
            out[start:stop] = panel
            start = stop

        stop += self._pos
        out[start:stop] = self._panels[-1][: self._pos]

        return out

    def clear(self):
        import numba

        initial = len(self._panels[0])
        self._panels = numba.typed.List([numpy.empty((initial,), dtype=self.dtype)])
        self._length = 0
        self._pos = 0


def _register():
    if hasattr(ak.numba, "ArrayViewType"):
        return

    import numba

    import awkward._connect.numba.arrayview
    import awkward._connect.numba.arrayview_cuda
    import awkward._connect.numba.builder
    import awkward._connect.numba.growablebuffer
    import awkward._connect.numba.layout
    import awkward._connect.numba.layoutbuilder

    n = ak.numba
    n.ArrayViewType = awkward._connect.numba.arrayview.ArrayViewType
    n.ArrayViewModel = awkward._connect.numba.arrayview.ArrayViewModel
    n.RecordViewType = awkward._connect.numba.arrayview.RecordViewType
    n.RecordViewModel = awkward._connect.numba.arrayview.RecordViewModel
    n.ContentType = awkward._connect.numba.layout.ContentType
    n.NumpyArrayType = awkward._connect.numba.layout.NumpyArrayType
    n.RegularArrayType = awkward._connect.numba.layout.RegularArrayType
    n.ListArrayType = awkward._connect.numba.layout.ListArrayType
    n.IndexedArrayType = awkward._connect.numba.layout.IndexedArrayType
    n.IndexedOptionArrayType = awkward._connect.numba.layout.IndexedOptionArrayType
    n.ByteMaskedArrayType = awkward._connect.numba.layout.ByteMaskedArrayType
    n.BitMaskedArrayType = awkward._connect.numba.layout.BitMaskedArrayType
    n.UnmaskedArrayType = awkward._connect.numba.layout.UnmaskedArrayType
    n.RecordArrayType = awkward._connect.numba.layout.RecordArrayType
    n.UnionArrayType = awkward._connect.numba.layout.UnionArrayType
    n.ArrayBuilderType = awkward._connect.numba.builder.ArrayBuilderType
    n.ArrayBuilderModel = awkward._connect.numba.builder.ArrayBuilderModel
    n.cuda = awkward._connect.numba.arrayview_cuda.ArrayViewArgHandler()
    n.GrowableBufferType = awkward._connect.numba.growablebuffer.GrowableBufferType
    n._from_data = awkward._connect.numba.growablebuffer._from_data

    @numba.extending.typeof_impl.register(ak.highlevel.Array)
    def typeof_Array(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak.highlevel.Record)
    def typeof_Record(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak.highlevel.ArrayBuilder)
    def typeof_ArrayBuilder(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.BitMasked)
    def typeof_layoutbuilder_BitMasked(val, c):
        return ak._connect.numba.layoutbuilder.BitMaskedType(
            val._mask._dtype,
            val._content,
            val._valid_when,
            val._lsb_order,
            val._parameters,
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.ByteMasked)
    def typeof_layoutbuilder_ByteMasked(val, c):
        return ak._connect.numba.layoutbuilder.ByteMaskedType(
            val._content,
            val._valid_when,
            val._parameters,
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Empty)
    def typeof_layoutbuilder_Empty(val, c):
        return ak._connect.numba.layoutbuilder.EmptyType(val._parameters)

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.IndexedOption)
    def typeof_layoutbuilder_IndexedOption(val, c):
        return ak._connect.numba.layoutbuilder.IndexedOptionType(
            numba.from_dtype(val.dtype), val._content, val._parameters
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.ListOffset)
    def typeof_layoutbuilder_ListOffset(val, c):
        return ak._connect.numba.layoutbuilder.ListOffsetType(
            val._offsets._dtype, val._content, val._parameters
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Numpy)
    def typeof_layoutbuilder_Numpy(val, c):
        return ak._connect.numba.layoutbuilder.NumpyType(
            val._data._dtype, val._parameters
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Record)
    def typeof_layoutbuilder_Record(val, c):
        return ak._connect.numba.layoutbuilder.RecordType(
            val._contents, val._fields, val._parameters
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Regular)
    def typeof_layoutbuilder_Regular(val, c):
        return ak._connect.numba.layoutbuilder.RegularType(
            val._content, val._size, val._parameters
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Tuple)
    def typeof_layoutbuilder_Tuple(val, c):
        return ak._connect.numba.layoutbuilder.TupleType(val._contents, val._parameters)

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Union)
    def typeof_layoutbuilder_Union(val, c):
        return ak._connect.numba.layoutbuilder.UnionType(
            val._tags._dtype,
            val._index._dtype,
            val._contents,
            val._parameters,
        )

    @numba.extending.typeof_impl.register(ak.numba.layoutbuilder.Unmasked)
    def typeof_layoutbuilder_Unmasked(val, c):
        return ak._connect.numba.layoutbuilder.UnmaskedType(
            val._content, val._parameters
        )
