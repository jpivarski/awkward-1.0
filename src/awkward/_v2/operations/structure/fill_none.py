# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def fill_none(array, value, axis=-1, highlevel=True, behavior=None):
    pass


#     """
#     Args:
#         array: Data in which to replace None with a given value.
#         value: Data with which to replace None.
#         axis (None or int): If None, replace all None values in the array
#             with the given value; if an int, The dimension at which this
#             operation is applied. The outermost dimension is `0`, followed
#             by `1`, etc., and negative values count backward from the
#             innermost: `-1` is the innermost  dimension, `-2` is the next
#             level up, etc.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Replaces missing values (None) with a given `value`.

#     For example, in the following `array`,

#         ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

#     The None values could be replaced with `0` by

#         >>> ak.fill_none(array, 0)
#         <Array [[1.1, 0, 2.2], [], [0, 3.3, 4.4]] type='3 * var * float64'>

#     The replacement value doesn't strictly need the same type as the
#     surrounding data. For example, the None values could also be replaced
#     by a string.

#         >>> ak.fill_none(array, "hi")
#         <Array [[1.1, 'hi', 2.2], ... ['hi', 3.3, 4.4]] type='3 * var * union[float64, s...'>

#     The list content now has a union type:

#         >>> ak.type(ak.fill_none(array, "hi"))
#         3 * var * union[float64, string]

#     The values could be floating-point numbers or strings.
#     """

#     arraylayout = ak.operations.convert.to_layout(
#         array, allow_record=True, allow_other=False
#     )
#     nplike = ak.nplike.of(arraylayout)

#     # Convert value type to appropriate layout
#     if (
#         isinstance(value, np.ndarray)
#         and issubclass(value.dtype.type, (np.bool_, np.number))
#         and len(value.shape) != 0
#     ):
#         valuelayout = ak.operations.convert.to_layout(
#             nplike.asarray(value)[np.newaxis], allow_record=False, allow_other=False
#         )
#     elif isinstance(value, (bool, numbers.Number, np.bool_, np.number)) or (
#         isinstance(value, np.ndarray)
#         and issubclass(value.dtype.type, (np.bool_, np.number))
#     ):
#         valuelayout = ak.operations.convert.to_layout(
#             nplike.asarray(value), allow_record=False, allow_other=False
#         )
#     elif (
#         isinstance(value, Iterable)
#         and not (
#             isinstance(value, (str, bytes))
#             or (ak._util.py27 and isinstance(value, ak._util.unicode))
#         )
#         or isinstance(value, (ak.highlevel.Record, ak.layout.Record))
#     ):
#         valuelayout = ak.operations.convert.to_layout(
#             value, allow_record=True, allow_other=False
#         )
#         if isinstance(valuelayout, ak.layout.Record):
#             valuelayout = valuelayout.array[valuelayout.at : valuelayout.at + 1]
#         elif len(valuelayout) == 0:
#             offsets = ak.layout.Index64(nplike.array([0, 0], dtype=np.int64))
#             valuelayout = ak.layout.ListOffsetArray64(offsets, valuelayout)
#         else:
#             valuelayout = ak.layout.RegularArray(valuelayout, len(valuelayout), 1)
#     else:
#         valuelayout = ak.operations.convert.to_layout(
#             [value], allow_record=False, allow_other=False
#         )

#     def maybe_fillna(layout):
#         if isinstance(layout, ak._util.optiontypes):
#             return layout.fillna(valuelayout)
#         else:
#             return layout

#     if axis is None:

#         def transform(layout, depth, posaxis):
#             return ak._util.transform_child_layouts(
#                 transform, maybe_fillna(layout), depth, posaxis
#             )

#     else:

#         def transform(layout, depth, posaxis):
#             posaxis = layout.axis_wrap_if_negative(posaxis)
#             if posaxis + 1 < depth:
#                 return layout

#             if posaxis + 1 == depth:
#                 layout = maybe_fillna(layout)

#             return ak._util.transform_child_layouts(transform, layout, depth, posaxis)

#     out = transform(arraylayout, 1, axis)

#     return ak._util.maybe_wrap_like(out, array, behavior, highlevel)
