# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.ListArray32)
@numba.extending.typeof_impl.register(awkward1.layout.ListArray64)
def typeof(val, c):
    return ListArrayType(numba.typeof(numpy.asarray(val.starts)), numba.typeof(numpy.asarray(val.stops)), numba.typeof(val.content), numba.typeof(val.id))

class ListArrayType(content.ContentType):
    def __init__(self, startstpe, stopstpe, contenttpe, idtpe):
        assert startstpe == stopstpe
        super(ListArrayType, self).__init__(name="ListArray{}Type({}, id={})".format(startstpe.dtype.bitwidth, contenttpe.name, idtpe.name))
        self.startstpe = startstpe
        self.contenttpe = contenttpe
        self.idtpe = idtpe

    @property
    def stopstpe(self):
        return self.startstpe

    @property
    def bitwidth(self):
        return self.startstpe.dtype.bitwidth

    @property
    def ndim(self):
        return 1 + self.contenttpe.ndim

    def getitem_int(self):
        return self.contenttpe

    def getitem_range(self):
        return self

    def getitem_tuple(self, wheretpe):
        nexttpe = ListArrayType(util.index64tpe, util.index64tpe, self, numba.none)
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        import awkward1._numba.listoffsetarray
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.getitem_next(tailtpe, isadvanced)
        elif isinstance(headtpe, numba.types.SliceType):
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, self.contenttpe.getitem_next(tailtpe, isadvanced), self.idtpe)
        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")
        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")
        elif isinstance(headtpe, numba.types.Array) and not isadvanced:
            if headtpe.ndim != 1:
                raise NotImplementedError("array.ndim != 1")
            contenttpe = self.contenttpe.carry().getitem_next(tailtpe, True)
            return awkward1._numba.listoffsetarray.ListOffsetArrayType(self.startstpe, contenttpe, self.idtpe)
        elif isinstance(headtpe, numba.types.Array):
            return self.contenttpe.getitem_next(tailtpe, True)
        else:
            raise AssertionError(headtpe)

    def carry(self):
        return self

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

@numba.extending.register_model(ListArrayType)
class ListArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("starts", fe_type.startstpe),
                   ("stops", fe_type.stopstpe),
                   ("content", fe_type.contenttpe)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(ListArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ListArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    startsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (starts_obj,))
    stopsarray_obj = c.pyapi.call_function_objargs(asarray_obj, (stops_obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.starts = c.pyapi.to_native_value(tpe.startstpe, startsarray_obj).value
    proxyout.stops = c.pyapi.to_native_value(tpe.stopstpe, stopsarray_obj).value
    proxyout.content = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(startsarray_obj)
    c.pyapi.decref(stopsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ListArrayType)
def box(tpe, val, c):
    if tpe.bitwidth == 32:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        ListArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListArray32))
    elif tpe.bitwidth == 64:
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index64))
        ListArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListArray64))
    else:
        raise AssertionError("unrecognized bitwidth")
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    startsarray_obj = c.pyapi.from_native_value(tpe.startstpe, proxyin.starts, c.env_manager)
    stopsarray_obj = c.pyapi.from_native_value(tpe.stopstpe, proxyin.stops, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    starts_obj = c.pyapi.call_function_objargs(Index_obj, (startsarray_obj,))
    stops_obj = c.pyapi.call_function_objargs(Index_obj, (stopsarray_obj,))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(startsarray_obj)
    c.pyapi.decref(stopsarray_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(ListArray_obj, (starts_obj, stops_obj, content_obj, id_obj))
        c.pyapi.decref(id_obj)
    else:
        out = c.pyapi.call_function_objargs(ListArray_obj, (starts_obj, stops_obj, content_obj))
    c.pyapi.decref(ListArray_obj)
    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    return out

@numba.extending.lower_builtin(len, ListArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    startslen = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.startstpe), (proxyin.starts,))
    return startslen

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.startstpe.dtype(tpe.startstpe, wheretpe), (proxyin.starts, whereval))
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.startstpe.dtype(tpe.stopstpe, wheretpe), (proxyin.stops, whereval))
    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxyslice.start = util.cast(context, builder, tpe.startstpe.dtype, numba.intp, start)
    proxyslice.stop = util.cast(context, builder, tpe.stopstpe.dtype, numba.intp, stop)
    proxyslice.step = context.get_constant(numba.intp, 1)

    outtpe = tpe.contenttpe.getitem_range()
    return tpe.contenttpe.lower_getitem_range(context, builder, outtpe(tpe.contenttpe, numba.types.slice2_type), (proxyin.content, proxyslice._getvalue()))

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    import awkward1._numba.identity

    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.startstpe(tpe.startstpe, wheretpe), (proxyin.starts, whereval))
    proxyout.stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.stopstpe(tpe.stopstpe, wheretpe), (proxyin.stops, whereval))
    proxyout.content = proxyin.content
    if tpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, tpe.idtpe, wheretpe, proxyin.id, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args

    wheretpe, whereval = util.preprocess_slicetuple(context, builder, wheretpe, whereval)
    nexttpe, nextval = util.wrap_for_slicetuple(context, builder, arraytpe, arrayval)

    outtpe = nexttpe.getitem_next(wheretpe, False)
    outval = nexttpe.lower_getitem_next(context, builder, nexttpe, wheretpe, nextval, whereval, None)

    return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, context.get_constant(numba.int64, 0)))

@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, ListArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, ListArrayType, type(numba.typeof(numpy.newaxis)))
def lower_getitem_other(context, builder, sig, args):
    rettpe, (arraytpe, wheretpe) = sig.return_type, sig.args
    arrayval, whereval = args
    wrappedtpe = numba.types.Tuple((wheretpe,))
    wrappedval = context.make_tuple(builder, wrappedtpe, (whereval,))
    return lower_getitem_tuple(context, builder, rettpe(arraytpe, wrappedtpe), (arrayval, wrappedval))

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    import awkward1._numba.listoffsetarray

    if len(wheretpe.types) == 0:
        return arrayval

    headtpe = wheretpe.types[0]
    tailtpe = numba.types.Tuple(wheretpe.types[1:])
    headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
    tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    lenstarts = util.arraylen(context, builder, arraytpe.startstpe, proxyin.starts, totpe=numba.int64)
    lenstops = util.arraylen(context, builder, arraytpe.stopstpe, proxyin.stops, totpe=numba.int64)
    lencontent = util.arraylen(context, builder, arraytpe.contenttpe, proxyin.content, totpe=numba.int64)

    with builder.if_then(builder.icmp_signed("<", lenstops, lenstarts), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ("len(stops) < len(starts)",))

    if isinstance(headtpe, numba.types.Integer):
        assert advanced is None
        if arraytpe.bitwidth == 64:
            kernel = cpu.kernels.awkward_listarray64_getitem_next_at_64
        elif arraytpe.bitwidth == 32:
            kernel = cpu.kernels.awkward_listarray32_getitem_next_at_64
        else:
            raise AssertionError("unrecognized bitwidth")

        nextcarry = util.newindex64(context, builder, numba.int64, lenstarts)
        util.call(context, builder, kernel,
            (util.arrayptr(context, builder, util.index64tpe, nextcarry),
             util.arrayptr(context, builder, arraytpe.startstpe, proxyin.starts),
             util.arrayptr(context, builder, arraytpe.stopstpe, proxyin.stops),
             lenstarts,
             context.get_constant(numba.int64, 0),
             context.get_constant(numba.int64, 0),
             util.cast(context, builder, headtpe, numba.int64, headval)),
            "in {}, indexing error".format(arraytpe.shortname))
        nextcontenttpe = arraytpe.contenttpe.carry()
        nextcontentval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)
        return nextcontenttpe.lower_getitem_next(context, builder, nextcontenttpe, tailtpe, nextcontentval, tailval, advanced)

    elif isinstance(headtpe, numba.types.SliceType):
        raise NotImplementedError("ListArray.getitem_next(slice)")

    elif isinstance(headtpe, numba.types.EllipsisType):
        raise NotImplementedError("ListArray.getitem_next(ellipsis)")

    elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
        raise NotImplementedError("ListArray.getitem_next(newaxis)")

    elif isinstance(headtpe, numba.types.Array):
        if headtpe.ndim != 1:
            raise NotImplementedError("array.ndim != 1")

        flathead = numba.targets.arrayobj.array_ravel(context, builder, util.index64tpe(headtpe), (headval,))
        lenflathead = util.arraylen(context, builder, util.index64tpe, flathead, totpe=numba.int64)

        if advanced is None:
            if arraytpe.bitwidth == 64:
                kernel = cpu.kernels.awkward_listarray64_getitem_next_array_64
            elif arraytpe.bitwidth == 32:
                kernel = cpu.kernels.awkward_listarray32_getitem_next_array_64
            else:
                raise AssertionError("unrecognized bitwidth")

            lencarry = builder.mul(lenstarts, lenflathead)
            lenoffsets = builder.add(lenstarts, context.get_constant(numba.int64, 1))

            nextcarry = util.newindex64(context, builder, numba.int64, lencarry)
            nextadvanced = util.newindex64(context, builder, numba.int64, lencarry)
            nextoffsets = util.newindex64(context, builder, numba.int64, lenoffsets)
            util.call(context, builder, kernel,
                (util.arrayptr(context, builder, util.index64tpe, nextoffsets),
                 util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, arraytpe.startstpe, proxyin.starts),
                 util.arrayptr(context, builder, arraytpe.stopstpe, proxyin.stops),
                 util.arrayptr(context, builder, util.index64tpe, flathead),
                 context.get_constant(numba.int64, 0),
                 context.get_constant(numba.int64, 0),
                 lenstarts,
                 lenflathead,
                 lencontent),
                "in {}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            contenttpe = nexttpe.getitem_next(tailtpe, True)
            contentval = nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

            outtpe = awkward1._numba.listoffsetarray.ListOffsetArrayType(arraytpe.startstpe, contenttpe, arraytpe.idtpe)
            proxyout = numba.cgutils.create_struct_proxy(outtpe)(context, builder)
            proxyout.offsets = nextoffsets
            proxyout.content = contentval
            if outtpe.idtpe != numba.none:
                proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, outtpe.idtpe, util.index64tpe, proxyin.id, flathead)
            return proxyout._getvalue()

        else:
            if arraytpe.bitwidth == 64:
                kernel = cpu.kernels.awkward_listarray64_getitem_next_array_advanced_64
            elif arraytpe.bitwidth == 32:
                kernel = cpu.kernels.awkward_listarray32_getitem_next_array_advanced_64
            else:
                raise AssertionError("unrecognized bitwidth")

            nextcarry = util.newindex64(context, builder, numba.int64, lenstarts)
            nextadvanced = util.newindex64(context, builder, numba.int64, lenstarts)
            util.call(context, builder, kernel,
                (util.arrayptr(context, builder, util.index64tpe, nextcarry),
                 util.arrayptr(context, builder, util.index64tpe, nextadvanced),
                 util.arrayptr(context, builder, arraytpe.startstpe, proxyin.starts),
                 util.arrayptr(context, builder, arraytpe.stopstpe, proxyin.stops),
                 util.arrayptr(context, builder, util.index64tpe, flathead),
                 util.arrayptr(context, builder, util.index64tpe, advanced),
                 context.get_constant(numba.int64, 0),
                 context.get_constant(numba.int64, 0),
                 lenstarts,
                 lenflathead,
                 lencontent),
                "in {}, indexing error".format(arraytpe.shortname))

            nexttpe = arraytpe.contenttpe.carry()
            nextval = arraytpe.contenttpe.lower_carry(context, builder, arraytpe.contenttpe, util.index64tpe, proxyin.content, nextcarry)

            outtpe = nexttpe.getitem_next(tailtpe, True)
            return nexttpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, nextadvanced)

    else:
        raise AssertionError(headtpe)

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    import awkward1._numba.identity

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)

    proxyout = numba.cgutils.create_struct_proxy(arraytpe)(context, builder)
    proxyout.starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.startstpe(arraytpe.startstpe, carrytpe), (proxyin.starts, carryval))
    proxyout.stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, arraytpe.stopstpe(arraytpe.stopstpe, carrytpe), (proxyin.stops, carryval))
    proxyout.content = proxyin.content
    if arraytpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, arraytpe.idtpe, carrytpe, proxyin.id, carryval)
    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = ListArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "starts":
            return tpe.startstpe

        elif attr == "stops":
            return tpe.stopstpe

        elif attr == "content":
            return tpe.contenttpe

        elif attr == "id":
            if tpe.idtpe == numba.none:
                return numba.optional(identity.IdentityType(numba.int32[:, :]))
            else:
                return tpe.idtpe

@numba.extending.lower_getattr(ListArrayType, "starts")
def lower_starts(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.startstpe, proxyin.starts)
    return proxyin.starts

@numba.extending.lower_getattr(ListArrayType, "stops")
def lower_stops(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.stopstpe, proxyin.stops)
    return proxyin.stops

@numba.extending.lower_getattr(ListArrayType, "content")
def lower_content(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe.contenttpe, proxyin.content)
    return proxyin.content

@numba.extending.lower_getattr(ListArrayType, "id")
def lower_id(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.idtpe == numba.none:
        return context.make_optional_none(builder, identity.IdentityType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.idtpe, proxyin.id)
        return proxyin.id
