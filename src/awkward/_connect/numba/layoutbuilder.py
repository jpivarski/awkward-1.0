# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import numba
import numba.core.typing.npydecl
import numpy as np

# from awkward._connect.numba.growablebuffer import GrowableBufferType
from awkward.numba.layoutbuilder import (
    BitMasked,
    ByteMasked,
    Empty,
    IndexedOption,
    ListOffset,
    Numpy,
    Record,
    Regular,
    Tuple,
    Union,
    Unmasked,
)


def to_numbatype(builder):
    if isinstance(builder, Numpy):
        return Numpy.numbatype(builder)
    elif isinstance(builder, Empty):
        return Empty.numbatype(builder)
    elif isinstance(builder, ListOffset):
        return ListOffset.numbatype(builder)
    elif isinstance(builder, Regular):
        return Regular.numbatype(builder)
    elif isinstance(builder, IndexedOption):
        return IndexedOption.numbatype(builder)
    elif isinstance(builder, ByteMasked):
        return ByteMasked.numbatype(builder)
    elif isinstance(builder, BitMasked):
        return BitMasked.numbatype(builder)
    elif isinstance(builder, Unmasked):
        return Unmasked.numbatype(builder)
    elif isinstance(builder, Record):
        return Record.numbatype(builder)
    elif isinstance(builder, Tuple):
        return Tuple.numbatype(builder)
    elif isinstance(builder, Union):
        return Union.numbatype(builder)
    elif isinstance(builder, tuple):
        return Tuple.numbatype(builder)
    else:
        return builder


class LayoutBuilderType(numba.types.Type):
    def _init(self, parameters):
        self._parameters = parameters

    @property
    def parameter(self, name):
        if name in self._parameters:
            return numba.types.StringLiteral(self._parameters[name])
        else:
            raise TypeError(f"LayoutBuilder.parameters does not have a {name!r}")

    @property
    def length(self):
        return numba.types.int64


@numba.extending.overload(len)
def LayoutBuilderType_len(builder):
    if isinstance(
        builder,
        (
            BitMaskedType,
            ByteMaskedType,
            EmptyType,
            IndexedOptionType,
            ListOffsetType,
            NumpyType,
            RecordType,
            RegularType,
            TupleType,
            UnionType,
            UnmaskedType,
        ),
    ):

        def len_impl(builder):
            return builder._length_get()

        return len_impl


########## Numpy ############################################################


class NumpyType(LayoutBuilderType):
    def __init__(self, dtype, parameters):
        super().__init__(name=f"ak.lb.Numpy({dtype!r}, parameters={parameters!r})")
        self._dtype = dtype
        self._init(parameters)

    @property
    def dtype(self):
        return self._dtype

    @property
    def data(self):
        return numba.types.ListType(self.dtype)


#
# @numba.extending.typeof_impl.register(NumpyType)
# def typeof_NumpyType(val, c):
#     return NumpyType(numba.from_dtype(val.dtype))
#


@numba.extending.register_model(NumpyType)
class NumpyModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("data",):
    numba.extending.make_attribute_wrapper(NumpyType, member, "_" + member)


@numba.extending.overload_attribute(NumpyType, "dtype")
def NumpyType_dtype(builder):
    def getter(builder):
        return builder._data._dtype

    return getter


@numba.extending.unbox(NumpyType)
def NumpyType_unbox(typ, obj, c):
    # get PyObjects
    data_obj = c.pyapi.object_getattr_string(obj, "_data")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.data = c.pyapi.to_native_value(typ.data, data_obj).value

    # decref PyObjects
    c.pyapi.decref(data_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(NumpyType)
def NumpyType_box(typ, val, c):
    # get PyObject of the Numpy class and _from_buffer constructor
    Numpy_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Numpy))
    from_buffer_obj = c.pyapi.object_getattr_string(Numpy_obj, "_from_buffer")

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    data_obj = c.pyapi.from_native_value(typ.data, builder.data, c.env_manager)

    out = c.pyapi.call_function_objargs(
        from_buffer_obj,
        (data_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Numpy_obj)
    c.pyapi.decref(from_buffer_obj)

    c.pyapi.decref(data_obj)

    return out


def _from_buffer():
    raise RuntimeError("_from_buffer Python function is only implemented in Numba")


@numba.extending.type_callable(_from_buffer)
def Numpy_from_buffer_typer(context):
    def typer(buffer):
        if isinstance(buffer, numba.types.ListType):
            return NumpyType(buffer.dtype, parameters=None)

    return typer


@numba.extending.lower_builtin(_from_buffer, numba.types.ListType)
def Numpy_from_buffer_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.data = args[0]

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    return out._getvalue()


@numba.extending.overload(Numpy)
def Numpy_ctor(dtype, parameters=None):
    if isinstance(dtype, numba.types.StringLiteral):
        dt = np.dtype(dtype.literal_value)

    elif isinstance(dtype, numba.types.DTypeSpec):
        dt = numba.core.typing.npydecl.parse_dtype(dtype)

    else:
        return

    def ctor_impl(dtype, parameters=None):
        data = numba.typed.List()
        data.append(dt(0))
        data.pop()
        return _from_buffer(data)

    return ctor_impl


@numba.extending.overload_method(NumpyType, "_length_get", inline="always")
def Numpy_length(builder):
    def getter(builder):
        return len(builder._data)

    return getter


@numba.extending.overload_attribute(NumpyType, "dtype", inline="always")
def Numpy_dtype(builder):
    def get(builder):
        return builder._data._dtype

    return get


@numba.extending.overload_attribute(NumpyType, "data", inline="always")
def Numpy_buffer(builder):
    def get(builder):
        return builder._data

    return get


@numba.extending.overload_method(NumpyType, "append")
def Numpy_append(builder, datum):
    if isinstance(builder, NumpyType):

        def append(builder, datum):
            builder.data.append(builder.data._dtype(datum))  # FIXME

        return append


@numba.extending.overload_method(NumpyType, "extend")
def Numpy_extend(builder, data):
    def extend(builder, data):
        for x in data:
            builder.data.append(x)

    return extend


########## Empty ############################################################


class EmptyType(LayoutBuilderType):
    def __init__(self, parameters):
        super().__init__(name=f"ak.lb.Empty(parameters={parameters!r})")
        self._init(parameters)


@numba.extending.register_model(EmptyType)
class EmptyModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = []
        super().__init__(dmm, fe_type, members)


@numba.extending.unbox(EmptyType)
def EmptyType_unbox(typ, obj, c):
    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(EmptyType)
def EmptyType_box(typ, val, c):
    # get PyObject of the Empty class
    Empty_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Empty))

    out = c.pyapi.call_function_objargs(Empty_obj, ())

    # decref PyObjects
    c.pyapi.decref(Empty_obj)

    return out


@numba.extending.overload_method(EmptyType, "_length_get", inline="always")
def Empty_length(builder):
    def getter(builder):
        return 0

    return getter


@numba.extending.overload_method(EmptyType, "append")
def Empty_append(builder, datum):
    if isinstance(builder, EmptyType):

        def append(builder, datum):
            raise NumbaTypeError("Empty cannot append data")

        return append


########## ListOffset #########################################################


class ListOffsetType(LayoutBuilderType):
    def __init__(self, dtype, content, parameters):
        super().__init__(
            name=f"ak.lb.ListOffset({dtype!r}, {content.numbatype()}, parameters={parameters!r})"
        )
        self._dtype = dtype
        self._content = content
        self._init(parameters)

    @property
    def dtype(self):
        return self._dtype

    @property
    def offsets(self):
        return numba.types.ListType(self.dtype)

    @property
    def content(self):
        return to_numbatype(self._content)


@numba.extending.register_model(ListOffsetType)
class ListOffsetModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("offsets", fe_type.offsets),
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "offsets",
    "content",
):
    numba.extending.make_attribute_wrapper(ListOffsetType, member, "_" + member)


@numba.extending.overload_attribute(ListOffsetType, "dtype")
def ListOffsetType_dtype(builder):
    def getter(builder):
        return builder._offsets._dtype

    return getter


@numba.extending.unbox(ListOffsetType)
def ListOffsetType_unbox(typ, obj, c):
    # get PyObjects
    offsets_obj = c.pyapi.object_getattr_string(obj, "_offsets")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.offsets = c.pyapi.to_native_value(typ.offsets, offsets_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(ListOffsetType)
def ListOffsetType_box(typ, val, c):
    # get PyObject of the ListOffset class
    ListOffset_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ListOffset))
    dtype_obj = c.pyapi.object_getattr_string(ListOffset_obj, "dtype")

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    offsets_obj = c.pyapi.from_native_value(typ.offsets, builder.offsets, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        ListOffset_obj,
        (
            dtype_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(ListOffset_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(ListOffsetType, "_length_get", inline="always")
def ListOffset_length(builder):
    def getter(builder):
        return len(builder._offsets) - 1

    return getter


@numba.extending.overload_method(ListOffsetType, "_offsets", inline="always")
def ListOffset_offsets(builder):
    def getter(builder):
        return builder._offsets

    return getter


@numba.extending.overload_method(ListOffsetType, "begin_list", inline="always")
def ListOffset_begin_list(builder):
    if isinstance(builder, ListOffsetType):

        def begin_list(builder):
            return builder._content

        return begin_list


@numba.extending.overload_method(ListOffsetType, "end_list", inline="always")
def ListOffset_end_list(builder):
    if isinstance(builder, ListOffsetType):

        def end_list(builder):
            builder._offsets.append(len(builder._content))

        return end_list


########## Regular ############################################################


class RegularType(LayoutBuilderType):
    def __init__(self, content, size, parameters):
        super().__init__(
            name=f"ak.lb.Regular({content.numbatype()}, {size}, parameters={parameters!r})"
        )
        self._content = content
        self._size = size
        self._init(parameters)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def size(self):
        return numba.types.int64


@numba.extending.register_model(RegularType)
class RegularModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("content", fe_type.content),
            ("size", fe_type.size),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "content",
    "size",
):
    numba.extending.make_attribute_wrapper(RegularType, member, "_" + member)


@numba.extending.unbox(RegularType)
def RegularType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    size_obj = c.pyapi.object_getattr_string(obj, "_size")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.size = c.pyapi.to_native_value(typ.size, size_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(RegularType)
def RegularType_box(typ, val, c):
    # get PyObject of the Regular class
    Regular_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Regular))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)
    size_obj = c.pyapi.from_native_value(typ.size, builder.size, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Regular_obj,
        (
            content_obj,
            size_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(Regular_obj)

    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)

    return out


@numba.extending.overload_method(RegularType, "_length_get", inline="always")
def Regular_length(builder):
    def getter(builder):
        return math.floor(len(builder._content) / builder._size)

    return getter


@numba.extending.overload_method(RegularType, "_size", inline="always")
def Regular_starts(builder):
    def getter(builder):
        return builder._size

    return getter


@numba.extending.overload_method(RegularType, "begin_list", inline="always")
def Regular_begin_list(builder):
    if isinstance(builder, RegularType):

        def begin_list(builder):
            return builder._content

        return begin_list


@numba.extending.overload_method(RegularType, "end_list", inline="always")
def Regular_end_list(builder):
    if isinstance(builder, RegularType):

        def end_list(builder):
            pass

        return end_list


########## IndexedOption #######################################################


class IndexedOptionType(LayoutBuilderType):
    def __init__(self, dtype, content, parameters):
        super().__init__(
            name=f"ak.lb.IndexedOption({dtype!r}, {content.numbatype()}, parameters={parameters!r})"
        )
        self._dtype = dtype
        self._content = content
        self._init(parameters)

    @property
    def dtype(self):
        return self._dtype

    @property
    def index(self):
        return numba.types.ListType(self.dtype)

    @property
    def content(self):
        return to_numbatype(self._content)


@numba.extending.register_model(IndexedOptionType)
class IndexedOptionModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("index", fe_type.index),
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "index",
    "content",
):
    numba.extending.make_attribute_wrapper(IndexedOptionType, member, "_" + member)


@numba.extending.overload_attribute(IndexedOptionType, "dtype")
def IndexedOptionType_dtype(builder):
    def getter(builder):
        return builder._index._dtype

    return getter


@numba.extending.unbox(IndexedOptionType)
def IndexedOptionType_unbox(typ, obj, c):
    # get PyObjects
    index_obj = c.pyapi.object_getattr_string(obj, "_index")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.index = c.pyapi.to_native_value(typ.index, index_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(index_obj)
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(IndexedOptionType)
def IndexedOptionType_box(typ, val, c):
    # get PyObject of the Indexed class
    IndexedOption_obj = c.pyapi.unserialize(c.pyapi.serialize_object(IndexedOption))
    dtype_obj = c.pyapi.object_getattr_string(IndexedOption_obj, "dtype")

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    index_obj = c.pyapi.from_native_value(typ.index, builder.index, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        IndexedOption_obj,
        (
            dtype_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(IndexedOption_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(IndexedOptionType, "_length_get", inline="always")
def IndexedOption_length(builder):
    def getter(builder):
        return len(builder._index)

    return getter


@numba.extending.overload_method(IndexedOptionType, "_index", inline="always")
def IndexedOption_index(builder):
    def getter(builder):
        return builder._index

    return getter


@numba.extending.overload_method(IndexedOptionType, "append_valid")
def IndexedOption_append_valid(builder):
    if isinstance(builder, IndexedOptionType):

        def append_valid(builder):
            builder._index.append(len(builder._content))
            return builder._content

        return append_valid


@numba.extending.overload_method(IndexedOptionType, "extend_valid")
def IndexedOption_extend_valid(builder, size):
    def extend_valid(builder, size):
        start = len(builder._content)
        stop = start + size
        for x in range(start, stop):
            builder._index.append(builder._index._dtype(x))
        return builder._content

    return extend_valid


@numba.extending.overload_method(IndexedOptionType, "append_invalid")
def IndexedOption_append_invalid(builder):
    if isinstance(builder, IndexedOptionType):

        def append_invalid(builder):
            builder._index.append(builder._index._dtype(-1))

        return append_invalid


@numba.extending.overload_method(IndexedOptionType, "extend_invalid")
def IndexedOption_extend_invalid(builder, size):
    def extend_invalid(builder, size):
        for _ in range(size):
            builder._index.append(builder._index._dtype(-1))

    return extend_invalid


########## ByteMasked #########################################################


class ByteMaskedType(LayoutBuilderType):
    def __init__(self, content, valid_when, parameters):
        super().__init__(
            name=f"ak.lb.ByteMasked({content.numbatype()}, valid_when={valid_when}, parameters={parameters!r})"
        )
        self._content = content
        self._valid_when = valid_when
        self._init(parameters)

    @property
    def valid_when(self):
        return numba.types.boolean

    @property
    def mask(self):
        return numba.types.ListType(numba.types.boolean)

    @property
    def content(self):
        return to_numbatype(self._content)


@numba.extending.register_model(ByteMaskedType)
class ByteMaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("mask", fe_type.mask),
            ("content", fe_type.content),
            ("valid_when", fe_type.valid_when),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "mask",
    "content",
    "valid_when",
):
    numba.extending.make_attribute_wrapper(ByteMaskedType, member, "_" + member)


@numba.extending.unbox(ByteMaskedType)
def ByteMaskedType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    mask_obj = c.pyapi.object_getattr_string(obj, "_mask")
    valid_when_obj = c.pyapi.object_getattr_string(obj, "_valid_when")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.mask = c.pyapi.to_native_value(typ.mask, mask_obj).value
    out.valid_when = c.pyapi.to_native_value(typ.valid_when, valid_when_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)
    c.pyapi.decref(mask_obj)
    c.pyapi.decref(valid_when_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(ByteMaskedType)
def ByteMaskedType_box(typ, val, c):
    # get PyObject of the ByteMasked class
    ByteMasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ByteMasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        ByteMasked_obj,
        (content_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(ByteMasked_obj)

    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(ByteMaskedType, "_length_get", inline="always")
def ByteMasked_length(builder):
    if isinstance(builder, ByteMaskedType):

        def getter(builder):
            return len(builder._content)

        return getter


@numba.extending.overload_method(ByteMaskedType, "append_valid")
def ByteMasked_append_valid(builder):
    if isinstance(builder, ByteMaskedType):

        def append_valid(builder):
            builder._mask.append(builder._valid_when)
            return builder._content

        return append_valid


@numba.extending.overload_method(ByteMaskedType, "extend_valid")
def ByteMasked_extend_valid(builder, size):
    if isinstance(builder, ByteMaskedType):

        def extend_valid(builder, size):
            builder._mask.extend([builder._valid_when] * size)
            return builder._content

        return extend_valid


@numba.extending.overload_method(ByteMaskedType, "append_invalid")
def ByteMasked_append_invalid(builder):
    if isinstance(builder, ByteMaskedType):

        def append_invalid(builder):
            builder._mask.append(not builder._valid_when)
            return builder._content

        return append_invalid


@numba.extending.overload_method(ByteMaskedType, "extend_invalid")
def ByteMasked_extend_invalid(builder, size):
    if isinstance(builder, ByteMaskedType):

        def extend_invalid(builder, size):
            builder._mask.extend([not builder._valid_when] * size)
            return builder._content

        return extend_invalid


########## BitMasked #########################################################


class BitMaskedType(LayoutBuilderType):
    def __init__(self, dtype, content, valid_when, lsb_order, parameters):
        super().__init__(
            name=f"ak.lb.BitMasked({dtype}, {content.numbatype()}, {valid_when}, {lsb_order}, parameters={parameters!r})"
        )
        self._dtype = dtype
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._init(parameters)

    @property
    def dtype(self):
        return self._dtype

    @property
    def mask(self):
        return numba.types.ListType(self.dtype)

    @property
    def valid_when(self):
        return numba.types.boolean

    @property
    def lsb_order(self):
        return numba.types.boolean

    @property
    def current_byte_index(self):
        return numba.types.Array(numba.types.uint8, 1, "C")

    @property
    def content(self):
        return to_numbatype(self._content)


@numba.extending.register_model(BitMaskedType)
class BitMaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("mask", fe_type.mask),
            ("content", fe_type.content),
            ("valid_when", fe_type.valid_when),
            ("lsb_order", fe_type.lsb_order),
            ("current_byte_index", fe_type.current_byte_index),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "mask",
    "content",
    "valid_when",
    "lsb_order",
    "current_byte_index",
):
    numba.extending.make_attribute_wrapper(BitMaskedType, member, "_" + member)


@numba.extending.overload_attribute(BitMaskedType, "dtype")
def BitMaskedType_dtype(builder):
    def getter(builder):
        return builder._mask._dtype

    return getter


@numba.extending.unbox(BitMaskedType)
def BitMaskedType_unbox(typ, obj, c):
    # get PyObjects
    mask_obj = c.pyapi.object_getattr_string(obj, "_mask")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    valid_when_obj = c.pyapi.object_getattr_string(obj, "_valid_when")
    lsb_order_obj = c.pyapi.object_getattr_string(obj, "_lsb_order")
    current_byte_index_obj = c.pyapi.object_getattr_string(obj, "_current_byte_index")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.mask = c.pyapi.to_native_value(typ.mask, mask_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.valid_when = c.pyapi.to_native_value(typ.valid_when, valid_when_obj).value
    out.lsb_order = c.pyapi.to_native_value(typ.lsb_order, lsb_order_obj).value
    out.current_byte_index = c.pyapi.to_native_value(
        typ.current_byte_index, current_byte_index_obj
    ).value

    # decref PyObjects
    c.pyapi.decref(mask_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(valid_when_obj)
    c.pyapi.decref(lsb_order_obj)
    c.pyapi.decref(current_byte_index_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(BitMaskedType)
def BitMaskedType_box(typ, val, c):
    # get PyObject of the BitMasked class
    BitMasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(BitMasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    mask_obj = c.pyapi.from_native_value(typ.mask, builder.mask, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)
    valid_when_obj = c.pyapi.from_native_value(
        typ.valid_when, builder.valid_when, c.env_manager
    )
    lsb_order_obj = c.pyapi.from_native_value(
        typ.lsb_order, builder.lsb_order, c.env_manager
    )

    out = c.pyapi.call_function_objargs(
        BitMasked_obj,
        (
            mask_obj,
            content_obj,
            valid_when_obj,
            lsb_order_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(BitMasked_obj)

    c.pyapi.decref(mask_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(valid_when_obj)
    c.pyapi.decref(lsb_order_obj)

    return out


@numba.extending.overload_attribute(BitMaskedType, "_cast")
def BitMaskedType_cast(builder):
    def get_cast(builder):
        if builder._lsb_order:
            return np.array(
                [
                    np.uint8(1 << 0),
                    np.uint8(1 << 1),
                    np.uint8(1 << 2),
                    np.uint8(1 << 3),
                    np.uint8(1 << 4),
                    np.uint8(1 << 5),
                    np.uint8(1 << 6),
                    np.uint8(1 << 7),
                ]
            )
        else:
            return np.array(
                [
                    np.uint8(128 >> 0),
                    np.uint8(128 >> 1),
                    np.uint8(128 >> 2),
                    np.uint8(128 >> 3),
                    np.uint8(128 >> 4),
                    np.uint8(128 >> 5),
                    np.uint8(128 >> 6),
                    np.uint8(128 >> 7),
                ]
            )

    return get_cast


@numba.extending.overload_method(BitMaskedType, "_length_get", inline="always")
def BitMasked_length(builder):
    def getter(builder):
        return len(builder._content)

    return getter


@numba.extending.overload_method(BitMaskedType, "_append_begin", inline="always")
def BitMasked_append_begin(builder):
    def append_begin(builder):
        if builder._current_byte_index[1] == np.uint8(8):
            builder._current_byte_index[0] = np.uint8(0)
            builder._mask.append(np.uint8(0))
            builder._current_byte_index[1] = np.uint8(0)

    return append_begin


@numba.extending.overload_method(BitMaskedType, "_append_end", inline="always")
def BitMasked_append_end(builder):
    def append_end(builder):
        builder._current_byte_index[1] += np.uint8(1)
        if builder._valid_when:
            # 0 indicates null, 1 indicates valid
            builder._mask._panels[-1][builder._mask._length_pos[1] - 1] = (
                builder._current_byte_index[0]
            )
        else:
            # 0 indicates valid, 1 indicates null
            builder._mask[-1] = np.uint8(~builder._current_byte_index[0])

    return append_end


@numba.extending.overload_method(BitMaskedType, "extend_valid")
def BitMasked_extend_valid(builder, size):
    if isinstance(builder, BitMaskedType):

        def extend_valid(builder, size):
            for _ in range(size):
                builder.append_valid()
            return builder._content

        return extend_valid


@numba.extending.overload_method(BitMaskedType, "append_valid")
def BitMasked_append_valid(builder):
    if isinstance(builder, BitMaskedType):

        def append_valid(builder):
            builder._append_begin()
            # current_byte_ and cast_: 0 indicates null, 1 indicates valid
            builder._current_byte_index[0] |= builder._cast[
                builder._current_byte_index[1]
            ]
            builder._append_end()
            return builder._content

        return append_valid


@numba.extending.overload_method(BitMaskedType, "append_invalid")
def BitMasked_append_invalid(builder):
    if isinstance(builder, BitMaskedType):

        def append_invalid(builder):
            builder._append_begin()
            # current_byte_ and cast_ default to null, no change
            builder._append_end()
            return builder._content

        return append_invalid


@numba.extending.overload_method(BitMaskedType, "extend_invalid")
def BitMasked_extend_invalid(builder, size):
    if isinstance(builder, BitMaskedType):

        def extend_invalid(builder, size):
            for _ in range(size):
                builder.append_invalid()
            return builder._content

        return extend_invalid


########## Unmasked #########################################################


class UnmaskedType(LayoutBuilderType):
    def __init__(self, content, parameters):
        super().__init__(
            name=f"ak.lb.Unmasked({content.numbatype()}, parameters={parameters!r})"
        )
        self._content = content

    @property
    def content(self):
        return to_numbatype(self._content)


@numba.extending.register_model(UnmaskedType)
class UnmaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("content",):
    numba.extending.make_attribute_wrapper(UnmaskedType, member, "_" + member)


@numba.extending.unbox(UnmaskedType)
def UnmaskedType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(UnmaskedType)
def UnmaskedType_box(typ, val, c):
    # get PyObject of the Unmasked class
    Unmasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Unmasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Unmasked_obj,
        (content_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Unmasked_obj)

    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(UnmaskedType, "_length_get", inline="always")
def Unmasked_length(builder):
    def getter(builder):
        return len(builder._content)

    return getter


########## Record #########################################################


class RecordType(LayoutBuilderType):
    def __init__(self, contents, fields, parameters):
        super().__init__(
            name=f"ak.lb.Record({contents}, {fields}, parameters={parameters!r})"
        )
        self._contents = contents
        self._fields = fields
        self._init(parameters)

    @property
    def contents(self):
        return numba.types.Tuple([to_numbatype(it) for it in self._contents])

    @property
    def fields(self):
        return numba.types.Tuple(
            to_numbatype([numba.types.StringLiteral(it) for it in self._fields])
        )


@numba.extending.register_model(RecordType)
class RecordModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("contents", fe_type.contents),
            ("fields", fe_type.fields),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "contents",
    "fields",
):
    numba.extending.make_attribute_wrapper(RecordType, member, "_" + member)


@numba.extending.unbox(RecordType)
def RecordType_unbox(typ, obj, c):
    # get PyObjects
    contents_obj = c.pyapi.object_getattr_string(obj, "_contents")
    fields_obj = c.pyapi.object_getattr_string(obj, "_fields")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.contents = c.pyapi.to_native_value(typ.contents, contents_obj).value
    out.fields = c.pyapi.to_native_value(typ.fields, fields_obj).value

    # decref PyObjects
    c.pyapi.decref(contents_obj)
    c.pyapi.decref(fields_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(RecordType)
def RecordType_box(typ, val, c):
    # get PyObject of the Record class
    Record_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Record))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    contents_obj = c.pyapi.from_native_value(
        typ.contents, builder.contents, c.env_manager
    )
    fields_obj = c.pyapi.from_native_value(typ.fields, builder.fields, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Record_obj,
        (
            contents_obj,
            fields_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(Record_obj)

    c.pyapi.decref(contents_obj)
    c.pyapi.decref(fields_obj)

    return out


@numba.extending.overload_method(RecordType, "_length_get", inline="always")
def Record_length(builder):
    def getter(builder):
        return len(builder._contents[0])

    return getter


@numba.extending.overload_method(RecordType, "_field_index", inline="always")
def Record_field_index(builder, name):
    if isinstance(builder, RecordType):

        def field_index(builder, name):
            return builder._fields.index(name)

        return field_index


@numba.extending.overload_method(RecordType, "content")
def Record_content(builder, field_name):
    if isinstance(builder, RecordType):
        if isinstance(field_name, numba.types.IntegerLiteral):
            which = field_name.literal_value

            def getter_int(builder, field_name):
                return builder._contents[which]

            return getter_int

        if isinstance(field_name, numba.types.StringLiteral):
            which = builder._fields.index(field_name.literal_value)

            def getter_str(builder, field_name):
                return builder._contents[which]

            return getter_str


########## Tuple #######################################################


class TupleType(LayoutBuilderType):
    def __init__(self, contents, parameters):
        super().__init__(name=f"ak.lb.Tuple({contents}, parameters={parameters!r})")
        self._contents = contents
        self._init(parameters)

    @property
    def contents(self):
        return numba.types.Tuple([to_numbatype(it) for it in self._contents])


@numba.extending.register_model(TupleType)
class TupleModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("contents", fe_type.contents),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("contents",):
    numba.extending.make_attribute_wrapper(TupleType, member, "_" + member)


@numba.extending.unbox(TupleType)
def TupleType_unbox(typ, obj, c):
    # get PyObjects
    contents_obj = c.pyapi.object_getattr_string(obj, "_contents")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.contents = c.pyapi.to_native_value(typ.contents, contents_obj).value

    # decref PyObjects
    c.pyapi.decref(contents_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(TupleType)
def TupleType_box(typ, val, c):
    # get PyObject of the Tuple class
    Tuple_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Tuple))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    contents_obj = c.pyapi.from_native_value(
        typ.contents, builder.contents, c.env_manager
    )

    out = c.pyapi.call_function_objargs(
        Tuple_obj,
        (contents_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Tuple_obj)

    c.pyapi.decref(contents_obj)

    return out


@numba.extending.overload_method(TupleType, "_length_get", inline="always")
def Tuple_length(builder):
    if isinstance(builder, TupleType):

        def getter(builder):
            return len(builder._contents[0])

        return getter


@numba.extending.overload_method(TupleType, "content")
def Tuple_content(builder, index):
    if isinstance(builder, TupleType) and isinstance(index, numba.types.Integer):

        def getter(builder, index):
            content = builder._contents[numba.literally(index)]

            return content

        return getter


########## Union #######################################################


class UnionType(LayoutBuilderType):
    def __init__(self, tags_dtype, index_dtype, contents, parameters):
        super().__init__(
            name=f"ak.lb.Union({tags_dtype}, {index_dtype}, {contents},  parameters={parameters!r})"
        )
        self._tags_dtype = tags_dtype
        self._index_dtype = index_dtype
        self._contents = contents
        self._init(parameters)

    @property
    def tags(self):
        return numba.types.ListType(self._tags_dtype)

    @property
    def index(self):
        return numba.types.ListType(self._index_dtype)

    @property
    def contents(self):
        return numba.types.Tuple([to_numbatype(it) for it in self._contents])


@numba.extending.register_model(UnionType)
class UnionModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("tags", fe_type.tags),
            ("index", fe_type.index),
            ("contents", fe_type.contents),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "tags",
    "index",
    "contents",
):
    numba.extending.make_attribute_wrapper(UnionType, member, "_" + member)


@numba.extending.unbox(UnionType)
def UnionType_unbox(typ, obj, c):
    # get PyObjects
    tags_obj = c.pyapi.object_getattr_string(obj, "_tags")
    index_obj = c.pyapi.object_getattr_string(obj, "_index")
    contents_obj = c.pyapi.object_getattr_string(obj, "_contents")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.tags = c.pyapi.to_native_value(typ.tags, tags_obj).value
    out.index = c.pyapi.to_native_value(typ.index, index_obj).value
    out.contents = c.pyapi.to_native_value(typ.contents, contents_obj).value

    # decref PyObjects
    c.pyapi.decref(tags_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(contents_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(UnionType)
def UnionType_box(typ, val, c):
    # get PyObject of the Tuple class
    Union_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Union))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    tags_obj = c.pyapi.from_native_value(typ.tags, builder.tags, c.env_manager)
    index_obj = c.pyapi.from_native_value(typ.index, builder.index, c.env_manager)
    contents_obj = c.pyapi.from_native_value(
        typ.contents, builder.contents, c.env_manager
    )

    out = c.pyapi.call_function_objargs(
        Union_obj,
        (
            tags_obj,
            index_obj,
            contents_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(Union_obj)

    c.pyapi.decref(tags_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(contents_obj)

    return out


@numba.extending.overload_method(UnionType, "_length_get", inline="always")
def Union_length(builder):
    if isinstance(builder, UnionType):

        def getter(builder):
            return len(builder._contents[0])

        return getter


@numba.extending.overload_method(UnionType, "_tags", inline="always")
def Union_tags(builder):
    def getter(builder):
        return builder._tags

    return getter


@numba.extending.overload_method(UnionType, "_index", inline="always")
def Union_index(builder):
    def getter(builder):
        return builder._index

    return getter


@numba.extending.overload_method(UnionType, "append_content")
def Union_append_content(builder, tag):
    if isinstance(builder, UnionType) and isinstance(tag, numba.types.Integer):

        def append_content(builder, tag):
            content = builder._contents[numba.literally(tag)]
            # FIXME: cast to avoid
            # numba.core.errors.NumbaTypeSafetyWarning: unsafe cast from int64 to int8. Precision may be lost.
            builder._tags.append(builder._tags._dtype(tag))
            builder._index.append(builder._index._dtype(len(content)))
            return content

        return append_content
