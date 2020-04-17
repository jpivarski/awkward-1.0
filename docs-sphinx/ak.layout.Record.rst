ak.layout.Record
----------------

Represents a single value from a :doc:`ak.layout.RecordArray`.

As this is a columnar representation, the Record contains a
:doc:`ak.layout.RecordArray`, rather than the other way around.
Its two fields are

   * ``array``: the :doc:`ak.layout.RecordArray` and
   * ``at``: the index posiion where this Record is found.

The Record shares a reference with its :doc:`ak.layout.RecordArray`;
it is not a copy (not even a shallow-copied node object).

A :doc:`ak.layout.Record` node has the following properties and methods.
See their definitions in :doc:`ak.layout.Content` and :doc:`ak.layout.RecordArray`
as a guide. They pass through to the underlying ``array`` with adjustments
in some cases because Record is a scalar.

ak.layout.Record.__init__
=========================

.. py:method:: ak.layout.Record.__init__(array, at)

ak.layout.Record.array
======================

.. py:attribute:: ak.layout.Record.array

ak.layout.Record.at
===================

.. py:attribute:: ak.layout.Record.at

ak.layout.Record.__getitem__
============================

.. py:method:: ak.layout.Record.__getitem__(where)

ak.layout.Record.__repr__
=========================

.. py:method:: ak.layout.Record.__repr__()

ak.layout.Record.field
======================

.. py:method:: ak.layout.Record.field(index)

ak.layout.Record.field
======================

.. py:method:: ak.layout.Record.field(key)

ak.layout.Record.fieldindex
===========================

.. py:method:: ak.layout.Record.fieldindex(key)

ak.layout.Record.fielditems
===========================

.. py:method:: ak.layout.Record.fielditems()

ak.layout.Record.fields
=======================

.. py:method:: ak.layout.Record.fields()

ak.layout.Record.haskey
=======================

.. py:method:: ak.layout.Record.haskey(key)

ak.layout.Record.keys
=====================

.. py:method:: ak.layout.Record.keys()

ak.layout.Record.parameter
==========================

.. py:method:: ak.layout.Record.parameter(arg0)

ak.layout.Record.purelist_parameter
===================================

.. py:method:: ak.layout.Record.purelist_parameter(arg0)

ak.layout.Record.setparameter
=============================

.. py:method:: ak.layout.Record.setparameter(arg0, arg1)

ak.layout.Record.simplify
=========================

.. py:method:: ak.layout.Record.simplify()

ak.layout.Record.tojson
=======================

.. py:method:: ak.layout.Record.tojson(pretty=False, maxdecimals=None)

ak.layout.Record.tojson
=======================

.. py:method:: ak.layout.Record.tojson(destination, pretty=False, maxdecimals=None, buffersize=65536)

ak.layout.Record.type
=====================

.. py:method:: ak.layout.Record.type(arg0)

ak.layout.Record.astuple
========================

.. py:attribute:: ak.layout.Record.astuple

ak.layout.Record.identities
===========================

.. py:attribute:: ak.layout.Record.identities

ak.layout.Record.identity
=========================

.. py:attribute:: ak.layout.Record.identity

ak.layout.Record.istuple
========================

.. py:attribute:: ak.layout.Record.istuple

ak.layout.Record.numfields
==========================

.. py:attribute:: ak.layout.Record.numfields

ak.layout.Record.parameters
===========================

.. py:attribute:: ak.layout.Record.parameters
