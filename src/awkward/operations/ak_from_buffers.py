# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import awkward as ak
from awkward._backends.dispatch import regularize_backend
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._nplikes.virtual import VirtualArray
from awkward._regularize import is_integer
from awkward.forms.form import index_to_dtype, regularize_buffer_key

__all__ = ("from_buffers",)

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@high_level_function()
def from_buffers(
    form,
    length,
    container,
    buffer_key="{form_key}-{attribute}",
    *,
    backend="cpu",
    byteorder="<",
    allow_noncanonical_form=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to reconstitute from named buffers.
        length (int): Length of the array. (The output of this function is always
            single-partition.)
        container (Mapping, such as dict): The str \u2192 Python buffers that
            represent the decomposed Awkward Array. This `container` is only
            assumed to have a `__getitem__` method that accepts strings as keys.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            as keyword arguments and returns a string to use as a key for a buffer
            in the `container`.
        backend (str): Library to use to generate values that are
            put into the new array. The default, cpu, makes NumPy
            arrays, which are in main memory (e.g. not GPU). If all the values in
            `container` have the same `backend` as this, they won't be copied.
        byteorder (`"<"`, `">"`): Endianness of buffers read from `container`.
            If the byteorder does not match the current system byteorder, the
            arrays will be copied.
        allow_noncanonical_form (bool): If True, non-canonical forms will be
            simplified to produce arrays with canonical layouts; otherwise,
            an exception will be thrown for such forms.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Reconstitutes an Awkward Array from a Form, length, and a collection of memory
    buffers, so that data can be losslessly read from file formats and storage
    devices that only map names to binary blobs (such as a filesystem directory).

    The first three arguments of this function are the return values of
    #ak.to_buffers, so a full round-trip is

        >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

    The `container` argument lets you specify your own Mapping, which might be
    an interface to some storage format or device (e.g. h5py). It's okay if
    the `container` dropped NumPy's `dtype` and `shape` information, leaving
    raw bytes, since `dtype` and `shape` can be reconstituted from the
    #ak.forms.NumpyForm.
    If the values of `container` are recognised as arrays by the given backend,
    a view over their existing data will be used, where possible.

    The `buffer_key` should be the same as the one used in #ak.to_buffers.

    When `allow_noncanonical_form` is set to True, this function readily accepts
    non-simplified forms, i.e. forms which will be simplified by Awkward Array
    into "canonical" representations, e.g. `option[option[...]]` → `option[...]`.
    Such forms can be produced by the low-level ArrayBuilder `snapshot()` method.
    Given that Awkward Arrays must have canonical layouts, it follows that
    invoking this function with `allow_noncanonical_form` may produce arrays
    whose forms differ to the input form.

    In order for a non-simplified form to be considered valid, it should be one
    that the #ak.contents.Content layout classes could produce iff. the
    simplification rules were removed.


    See #ak.to_buffers for examples.
    """
    return _impl(
        form,
        length,
        container,
        buffer_key,
        backend,
        byteorder,
        highlevel,
        behavior,
        attrs,
        allow_noncanonical_form,
    )


def _impl(
    form,
    length,
    container,
    buffer_key,
    backend,
    byteorder,
    highlevel,
    behavior,
    attrs,
    simplify,
):
    backend = regularize_backend(backend)

    if isinstance(form, str):
        if ak.types.numpytype.is_primitive(form):
            form = ak.forms.NumpyForm(form)
        else:
            form = ak.forms.from_json(form)
    elif isinstance(form, dict):
        form = ak.forms.from_dict(form)

    if not (is_integer(length) and length >= 0):
        raise TypeError("'length' argument must be a non-negative integer")
    length = int(length)

    if not isinstance(form, ak.forms.Form):
        raise TypeError(
            "'form' argument must be a Form or its Python dict/JSON string representation"
        )

    getkey = regularize_buffer_key(buffer_key)

    out = _reconstitute(
        form, length, container, getkey, backend, byteorder, simplify, field_path=()
    )

    return wrap_layout(out, highlevel=highlevel, attrs=attrs, behavior=behavior)


def _from_buffer(
    nplike: NumpyLike,
    buffer,
    dtype: np.dtype,
    count: ShapeItem,
    byteorder: str,
    field_path: tuple,
) -> ArrayLike:
    # Unknown-length information implies that we didn't load shape-buffers (offsets, etc)
    # for the parent of this node. Thus, this node and its children *must* only
    # contain placeholders
    if count is unknown_length:
        # We may actually have a known buffer here, but as we do not know the length,
        # we cannot safely trim it. Thus, introduce a placeholder anyway
        return PlaceholderArray(nplike, (unknown_length,), dtype, field_path)
    # Known-length information implies that we should have known-length buffers here
    # We could choose to make this an error, and have the caller re-implement some
    # of #ak.from_buffers, or we can just introduce the known lengths where possible
    elif isinstance(buffer, PlaceholderArray) and buffer.size is unknown_length:
        return PlaceholderArray(nplike, (count,), dtype, field_path)
    elif isinstance(buffer, PlaceholderArray) or nplike.is_own_array(buffer):
        # Require 1D buffers
        array = nplike.reshape(buffer.view(dtype), shape=(-1,), copy=False)

        if array.size < count:
            raise TypeError(
                f"size of array ({array.size}) is less than size of form ({count})"
            )

        return array[:count]
    elif isinstance(buffer, VirtualArray):
        # we pass VirtualArray through as they are materialized later when needed.
        #
        # TODO:
        # What should we do though with the byteorder?
        # We could wrap the virtual array's generator return value with `nplike.from_buffer`
        # and `ak._util.native_to_byteorder` ...
        return buffer
    else:
        array = nplike.frombuffer(buffer, dtype=dtype, count=count)
        return ak._util.native_to_byteorder(array, byteorder)


def _reconstitute(
    form, length, container, getkey, backend, byteorder, simplify, field_path
):
    if isinstance(form, ak.forms.EmptyForm):
        if length != 0:
            raise ValueError(f"EmptyForm node, but the expected length is {length}")
        return ak.contents.EmptyArray()

    elif isinstance(form, ak.forms.NumpyForm):
        dtype = ak.types.numpytype.primitive_to_dtype(form.primitive)
        raw_array = container[getkey(form, "data")]
        real_length = length * math.prod(form.inner_shape)
        data = _from_buffer(
            backend.nplike,
            raw_array,
            dtype=dtype,
            count=real_length,
            byteorder=byteorder,
            field_path=field_path,
        )
        if form.inner_shape != ():
            data = backend.nplike.reshape(data, (length, *form.inner_shape))

        return ak.contents.NumpyArray(
            data, parameters=form._parameters, backend=backend
        )

    elif isinstance(form, ak.forms.UnmaskedForm):
        content = _reconstitute(
            form.content,
            length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        if simplify:
            make = ak.contents.UnmaskedArray.simplified
        else:
            make = ak.contents.UnmaskedArray
        return make(content, parameters=form._parameters)

    elif isinstance(form, ak.forms.BitMaskedForm):
        raw_array = container[getkey(form, "mask")]
        if length is unknown_length:
            next_length = unknown_length
        else:
            next_length = int(math.ceil(length / 8.0))
        mask = _from_buffer(
            backend.index_nplike,
            raw_array,
            dtype=index_to_dtype[form.mask],
            count=next_length,
            byteorder=byteorder,
            field_path=field_path,
        )
        content = _reconstitute(
            form.content,
            length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        if simplify:
            make = ak.contents.BitMaskedArray.simplified
        else:
            make = ak.contents.BitMaskedArray
        return make(
            ak.index.Index(mask),
            content,
            form.valid_when,
            length,
            form.lsb_order,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ByteMaskedForm):
        raw_array = container[getkey(form, "mask")]
        mask = _from_buffer(
            backend.index_nplike,
            raw_array,
            dtype=index_to_dtype[form.mask],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        content = _reconstitute(
            form.content,
            length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        if simplify:
            make = ak.contents.ByteMaskedArray.simplified
        else:
            make = ak.contents.ByteMaskedArray
        return make(
            ak.index.Index(mask),
            content,
            form.valid_when,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.IndexedOptionForm):
        raw_array = container[getkey(form, "index")]
        index = _from_buffer(
            backend.index_nplike,
            raw_array,
            dtype=index_to_dtype[form.index],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        if isinstance(index, PlaceholderArray):
            next_length = unknown_length
        else:
            next_length = (
                0 if len(index) == 0 else max(0, backend.index_nplike.max(index) + 1)
            )
        content = _reconstitute(
            form.content,
            next_length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        if simplify:
            make = ak.contents.IndexedOptionArray.simplified
        else:
            make = ak.contents.IndexedOptionArray
        return make(
            ak.index.Index(index),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.IndexedForm):
        raw_array = container[getkey(form, "index")]
        index = _from_buffer(
            backend.index_nplike,
            raw_array,
            dtype=index_to_dtype[form.index],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        if isinstance(index, PlaceholderArray):
            next_length = unknown_length
        else:
            next_length = (
                0
                if len(index) == 0
                else backend.index_nplike.index_as_shape_item(
                    backend.index_nplike.max(index) + 1
                )
            )
        content = _reconstitute(
            form.content,
            next_length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        if simplify:
            make = ak.contents.IndexedArray.simplified
        else:
            make = ak.contents.IndexedArray
        return make(
            ak.index.Index(index),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ListForm):
        raw_array1 = container[getkey(form, "starts")]
        raw_array2 = container[getkey(form, "stops")]
        starts = _from_buffer(
            backend.index_nplike,
            raw_array1,
            dtype=index_to_dtype[form.starts],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        stops = _from_buffer(
            backend.index_nplike,
            raw_array2,
            dtype=index_to_dtype[form.stops],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        if isinstance(stops, PlaceholderArray):
            next_length = unknown_length
        else:
            reduced_stops = stops[starts != stops]
            next_length = (
                0 if len(starts) == 0 else backend.index_nplike.max(reduced_stops)
            )
        content = _reconstitute(
            form.content,
            next_length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        return ak.contents.ListArray(
            ak.index.Index(starts),
            ak.index.Index(stops),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.ListOffsetForm):
        raw_array = container[getkey(form, "offsets")]
        offsets = _from_buffer(
            backend.index_nplike,
            raw_array,
            dtype=index_to_dtype[form.offsets],
            count=length + 1,
            byteorder=byteorder,
            field_path=field_path,
        )

        if isinstance(offsets, PlaceholderArray):
            next_length = unknown_length
        else:
            next_length = 0 if len(offsets) == 1 else offsets[-1]
        content = _reconstitute(
            form.content,
            next_length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        return ak.contents.ListOffsetArray(
            ak.index.Index(offsets),
            content,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.RegularForm):
        next_length = length * form.size
        content = _reconstitute(
            form.content,
            next_length,
            container,
            getkey,
            backend,
            byteorder,
            simplify,
            field_path,
        )
        return ak.contents.RegularArray(
            content,
            form.size,
            length,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.RecordForm):
        contents = [
            _reconstitute(
                content,
                length,
                container,
                getkey,
                backend,
                byteorder,
                simplify,
                (*field_path, field),
            )
            for content, field in zip(form.contents, form.fields)
        ]
        return ak.contents.RecordArray(
            contents,
            None if form.is_tuple else form.fields,
            length,
            parameters=form._parameters,
        )

    elif isinstance(form, ak.forms.UnionForm):
        raw_array1 = container[getkey(form, "tags")]
        raw_array2 = container[getkey(form, "index")]
        tags = _from_buffer(
            backend.index_nplike,
            raw_array1,
            dtype=index_to_dtype[form.tags],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        index = _from_buffer(
            backend.index_nplike,
            raw_array2,
            dtype=index_to_dtype[form.index],
            count=length,
            byteorder=byteorder,
            field_path=field_path,
        )
        if isinstance(index, PlaceholderArray) or isinstance(tags, PlaceholderArray):
            lengths = [unknown_length] * len(form.contents)
        else:
            lengths = []
            for tag in range(len(form.contents)):
                selected_index = index[tags == tag]
                if len(selected_index) == 0:
                    lengths.append(0)
                else:
                    lengths.append(backend.index_nplike.max(selected_index) + 1)
        contents = [
            _reconstitute(
                content,
                lengths[i],
                container,
                getkey,
                backend,
                byteorder,
                simplify,
                field_path,
            )
            for i, content in enumerate(form.contents)
        ]
        if simplify:
            make = ak.contents.UnionArray.simplified
        else:
            make = ak.contents.UnionArray
        return make(
            ak.index.Index(tags),
            ak.index.Index(index),
            contents,
            parameters=form._parameters,
        )

    else:
        raise AssertionError("unexpected form node type: " + str(type(form)))
