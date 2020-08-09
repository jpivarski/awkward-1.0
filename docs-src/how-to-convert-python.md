---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.10'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to convert to/from Python objects
=====================================

Builtin Python objects like dicts and lists can be converted into Awkward Arrays, and all Awkward Arrays can be converted into Python objects. Awkward type information, such as the distinction between fixed-size and variable-length lists, is lost in the transformation to Python objects.

```{code-cell} ipython3
import awkward1 as ak
import numpy as np
```

From Python to Awkward
----------------------

The function for Python → Awkward conversion is [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html).

```{code-cell} ipython3
py_objects = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
py_objects
```

```{code-cell} ipython3
ak_array = ak.from_iter(py_objects)
ak_array
```

See the sections below for how Python types are mapped to Awkward types.

Note that this should be considered a slow, memory-intensive function: not only does it need to iterate over Python data, but it needs to discover the type of the data progressively. Internally, this function uses an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) to accumulate data and discover types simultaneously. Don't, for instance, convert a large, numerical dataset from NumPy or Arrow into Python objects just to use [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html). There are specialized functions for that: see their tutorials (left-bar or ≡ button on mobile).

This is also the fallback operation of the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) and [ak.Record](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Record.html) constructors. Usually, small examples are built by passing Python objects directly to these constructors.

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

```{code-cell} ipython3
ak.Record({"x": 1, "y": [1.1, 2.2]})
```

From Awkward to Python
----------------------

The function for Awkward → Python conversion is [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html).

```{code-cell} ipython3
ak_array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
ak_array
```

```{code-cell} ipython3
ak.to_list(ak_array)
```

```{code-cell} ipython3
ak_record = ak.Record({"x": 1, "y": [1.1, 2.2]})
ak_record
```

```{code-cell} ipython3
ak.to_list(ak_record)
```

Note that this should be considered a slow, memory-intensive function, like [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html). Don't, for instance, convert a large, numerical dataset with [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) just to convert those lists into NumPy or Arrow. There are specialized functions for that: see their tutorials (left-bar or ≡ button on mobile).

Awkward Arrays and Records have a `tolist` method (note: no underscore), which is an analogy of [NumPy's tolist](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html). For small datasets (or a small slice of a dataset), this is a convenient way to get a quick view.

```{code-cell} ipython3
x = ak.Array(np.arange(1000))
y = ak.Array(np.tile(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), 100))
ak_array = ak.zip({"x": x, "y": y})
ak_array
```

```{code-cell} ipython3
ak_array[100].tolist()
```

```{code-cell} ipython3
ak_array[100:110].tolist()
```

Conversion of numbers and booleans
----------------------------------

Python `float`, `int`, and `bool` (so-called "primitive" types) are converted to `float64`, `int64`, and `bool` types in Awkward Arrays.

All floating-point Awkward types are converted to Python's `float`, all integral Awkward types are converted to Python's `int`, and Awkward's boolean type is converted to Python's `bool`.

```{code-cell} ipython3
ak.Array([1.1, 2.2, 3.3])
```

```{code-cell} ipython3
ak.Array([1.1, 2.2, 3.3]).tolist()
```

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5])
```

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5]).tolist()
```

```{code-cell} ipython3
ak.Array([True, False, True, False, False])
```

```{code-cell} ipython3
ak.Array([True, False, True, False, False]).tolist()
```

Conversion of lists
-------------------

Python lists, as well as iterables other than dict, tuple, str, and bytes, are converted to Awkward's variable-length lists. It is not possible to construct fixed-size lists with [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html). (One way to do that is by converting a NumPy array with [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html).)

Awkward's variable-length and fixed-size lists are converted into Python lists with [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html).

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
```

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).tolist()
```

```{code-cell} ipython3
ak.Array([[1, 2, 3], [4, 5, 6]])
```

```{code-cell} ipython3
ak.Array([[1, 2, 3], [4, 5, 6]]).tolist()
```

```{note}
Advanced topic: the rest of this section may be skipped if you don't care about the distinction between fixed-size and variable-length lists.
```

Note that a NumPy array is an iterable, so [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) iterates over it, constructing variable-length Awkward lists. By contrast, [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) casts the data (without iteration) into fixed-size Awkward lists.

```{code-cell} ipython3
np_array = np.array([[100, 200], [101, 201], [103, 203]])
np_array
```

```{code-cell} ipython3
ak.from_iter(np_array)
```

```{code-cell} ipython3
ak.from_numpy(np_array)
```

Note that the types differ: `var * int64` vs `2 * int64`. The [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor uses [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html) if given a NumPy array (with `dtype != "O"`) and [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) if given an iterable that it does not recognize.

This can be particularly subtle when NumPy arrays are nested within iterables.

```{code-cell} ipython3
np_array = np.array([[100, 200], [101, 201], [103, 203]])
np_array
```

```{code-cell} ipython3
# This is a NumPy array: constructor uses ak.from_numpy to get an array of fixed-size lists.
ak.Array(np_array)
```

```{code-cell} ipython3
py_objects = [np.array([100, 200]), np.array([101, 201]), np.array([103, 203])]
py_objects
```

```{code-cell} ipython3
# This is a list that contains NumPy arrays: constructor uses ak.from_iter to get an array of variable-length lists.
ak.Array(py_objects)
```

```{code-cell} ipython3
np_array_dtype_O = np.array([[100, 200], [101, 201], [103, 203]], dtype="O")
np_array_dtype_O
```

```{code-cell} ipython3
# This NumPy array has dtype="O", so it cannot be cast without iteration: constructor uses ak.from_iter.
ak.Array(np_array_dtype_O)
```

The logic behind this policy is that only NumPy arrays with `dtype != "O"` are guaranteed to have fixed-size contents. Other cases must have `var` type lists.

```{code-cell} ipython3
py_objects = [np.array([1.1, 2.2, 3.3]), np.array([]), np.array([4.4, 5.5])]
py_objects
```

```{code-cell} ipython3
ak.Array(py_objects)
```

```{code-cell} ipython3
np_array_dtype_O = np.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], dtype="O")
np_array_dtype_O
```

```{code-cell} ipython3
ak.Array(np_array_dtype_O)
```

Conversion of strings and bytestrings
-------------------------------------

Python strings (type `str`) are converted to and from Awkward's UTF-8 encoded strings and Python bytestrings (type `bytes`) are converted to and from Awkward's unencoded bytestrings.

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"])
```

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]).tolist()
```

```{code-cell} ipython3
ak.Array([b"one", b"two", b"three", b"four"])
```

```{code-cell} ipython3
ak.Array([b"one", b"two", b"three", b"four"]).tolist()
```

```{note}
Advanced topic: the rest of this section may be skipped if you don't care about internal representations.
```

Awkward's strings and bytestrings are not distinct types, but specializations of variable-length lists. Whereas a list might be internally represented by a [ListArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) or a [ListOffsetArray](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html),

```{code-cell} ipython3
ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
```

Strings and bytestrings are just [ListArrays](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListArray.html) and [ListOffsetArrays](https://awkward-array.readthedocs.io/en/latest/ak.layout.ListOffsetArray.html) of one-byte integers with special parameters:

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]).layout
```

These parameters indicate that the arrays of strings should have special behaviors, such as equality-per-string, rather than equality-per-character.

```{code-cell} ipython3
ak.Array([[1.1, 2.2], [], [3.3]]) == ak.Array([[1.1, 200], [], [3.3]])
```

```{code-cell} ipython3
ak.Array(["one", "two", "three", "four"]) == ak.Array(["one", "TWO", "thirty three", "four"])
```

(Without this overloaded behavior, the string comparison would yield `[True, True, True]` for `"one" == "one"` and would fail to broadcast `"three"` and `"thirty three"`.)

Special behaviors for strings are implemented using the same [ak.behavior](https://awkward-array.readthedocs.io/en/latest/ak.behavior.html) mechanism that you might use to give special behaviors to Arrays and Records.

```{code-cell} ipython3
ak.behavior["string"]
```

```{code-cell} ipython3
ak.behavior["bytestring"]
```

```{code-cell} ipython3
ak.behavior[np.equal, "string", "string"]
```

The fact that strings are really just variable-length lists is worth keeping in mind, since they might behave in unexpectedly list-like ways. If you notice any behavior that ought to be overloded for strings, recommend it as a [feature request](https://github.com/scikit-hep/awkward-1.0/issues/new?assignees=&labels=feature&template=feature-request.md&title=).

+++

Conversion of dicts and tuples
------------------------------

Python dicts with string-valued keys are converted to and from Awkward's record type with named fields. The data associated with different fields can have different types, but you generally want data associated with all instances of the same field to have the same type. Python dicts with non-string valued keys have no equivalent in Awkward Array (records are very different from mappings).

Python tuples are converted to and from Awkward's record type with unnamed fields. Note that Awkward views Python's lists and tuples in very different ways: lists are expected to be variable-length with all elements having the same type, while tuples are expected to be fixed-size with elements having potentially different types, just like a record.

In the following example, the `"x"` field has type `int64` and the `"y"` field has type `var * int64`.

```{code-cell} ipython3
ak_array_rec = ak.Array([{"x": 1, "y": [1, 2]}, {"x": 2, "y": []}])
ak_array_rec
```

```{code-cell} ipython3
ak_array_rec.tolist()
```

Here is the corresponding example with tuples:

```{code-cell} ipython3
ak_array_tup = ak.Array([(1, [1, 2]), (2, [])])
ak_array_tup
```

```{code-cell} ipython3
ak_array_tup.tolist()
```

Both of these Awkward types, `{"x": int64, "y": var * int64}` and `(int64, var * int64)`, have two fields, but the first one has names for those fields.

Both can be extracted using strings between square brackets, though the strings must be `"0"` and `"1"` for the tuple.

```{code-cell} ipython3
ak_array_rec["y"]
```

```{code-cell} ipython3
ak_array_rec["y", 1]
```

```{code-cell} ipython3
ak_array_tup["1"]
```

```{code-cell} ipython3
ak_array_tup["1", 1]
```

Note the difference in meaning between the `"1"` and the `1` in the above example. For safety, you may want to use the `slot[0-9]` methods with tuples:

```{code-cell} ipython3
ak_array_tup.slot1
```

```{code-cell} ipython3
ak_array_tup.slot1[1]
```

Or possibly [ak.unzip](https://awkward-array.readthedocs.io/en/latest/_auto/ak.unzip.html):

```{code-cell} ipython3
x, y = ak.unzip(ak_array_rec)
y
```

```{code-cell} ipython3
slot0, slot1 = ak.unzip(ak_array_tup)
slot1
```

That way, you can name the variables anything you like.

If fields are missing from some records, the missing values are filled in with None (option type: more on that below).

```{code-cell} ipython3
ak.Array([{"x": 1, "y": [1, 2]}, {"x": 2}])
```

If some tuples have different lengths, the resulting Awkward Array is taken to be heterogeneous (union type: more on that below).

```{code-cell} ipython3
ak.Array([(1, [1, 2]), (2,)])
```

An Awkward Record is a scalar drawn from a record array, so an [ak.Record](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Record.html) can be built from a single dict with string-valued keys.

```{code-cell} ipython3
ak.Record({"x": 1, "y": [1, 2], "z": 3.3})
```

The same is not true for tuples. The [ak.Record](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Record.html) constructor expects named fields.

```{code-cell} ipython3
:tags: [raises-exception]

ak.Record((1, [1, 2], 3.3))
```

Missing values: Python None
---------------------------

Python's None can appear anywhere in the structure parsed by [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html). It makes all data at that level of nesting have option type and is represented in [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) as None.

```{code-cell} ipython3
ak.Array([1.1, 2.2, None, 3.3, None, 4.4])
```

```{code-cell} ipython3
ak.Array([1.1, 2.2, None, 3.3, None, 4.4]).tolist()
```

```{note}
Advanced topic: the rest of this section describes the equivalence of missing record fields and record fields with None values, which is only relevant to datasets with missing fields.
```

As described above, fields that are absent from some records but not others are filled in with None. As a consequence, conversions from Python to Awkward Array back to Python don't necessarily result in the original expression:

```{code-cell} ipython3
ak.Array([
    {"x": 1.1, "y": [1]                    },
    {"x": 2.2,                 "z": "two"  },
    {"x": 3.3, "y": [1, 2, 3], "z": "three"}
]).tolist()
```

This is a deliberate choice. It would have been possible to convert records with missing fields into arrays with union type (more on that below), for which [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) would result in the original expression,

```{code-cell} ipython3
ak.concatenate([
    ak.Array([{"x": 1.1, "y": [1]                    }]),
    ak.Array([{"x": 2.2,                 "z": "two"  }]),
    ak.Array([{"x": 3.3, "y": [1, 2, 3], "z": "three"}]),
]).tolist()
```

But typical datasets of records with different sets of fields represent missing fields, rather than entirely different types of objects. (Even in particle physics applications that mix "electron objects" with "photon objects," both types of objects have the same trajectory fields `"x"`, `"y"`, `"z"` and differ in fields that exist for one and not the other, such as `"charge"` for electrons but not photons.)

The memory use of union arrays scales with the number of different types, up to $2^n$ for records with $n$ potentially missing fields. Option types of completely disjoint records with $n_1$ and $n_2$ fields use a memory footprint that scales as $n_1 + n_2$. Assuming that disjoint records are a single record type with missing fields is a recoverable mistake, but assuming that a single record type with missing fields are distinct for every combination of missing fields is potentially disastrous.

Tuples of different lengths, on the other hand, are assumed to be different types because mistaking slot $i$ for slot $i + 1$ would create unions anyway.

```{code-cell} ipython3
ak.Array([
    (1.1, [1]               ),
    (2.2,            "two"  ),
    (3.3, [1, 2, 3], "three"),
]).tolist()
```

Union types: heterogeneous data
-------------------------------

If the data in a Python iterable have different types at the same level of nesting ("heterogeneous"), the Awkward Arrays produced by [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) have union types.

Most Awkward operations are defined on union typed Arrays, but they're not generally not as efficient as the same operations on simply typed Arrays.

The following example mixes numbers (`float64`) with lists (`var * int64`).

```{code-cell} ipython3
ak.Array([1.1, 2.2, [], [1], [1, 2], 3.3])
```

The [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) function converts it back into a heterogeneous Python list.

```{code-cell} ipython3
ak.Array([1.1, 2.2, [], [1], [1, 2], 3.3]).tolist()
```

Any types may be mixed: numbers and lists, lists and records, missing data, etc.

```{code-cell} ipython3
ak.Array([[1, 2, 3], {"x": 1, "y": 2}, None])
```

One exception is that numerical data are merged without creating a union type: integers are expanded to floating point numbers.

```{code-cell} ipython3
ak.Array([1, 2, 3, 4, 5.5, 6.6, 7.7, 8, 9])
```

But booleans are not merged with integers.

```{code-cell} ipython3
ak.Array([1, 2, 3, True, True, False, 4, 5])
```

As described above, records with different sets of fields are presumed to be a single record type with missing values.

```{code-cell} ipython3
ak.type(ak.Array([
    {"x": 1.1, "y": [1]                    },
    {"x": 2.2,                 "z": "two"  },
    {"x": 3.3, "y": [1, 2, 3], "z": "three"}
]))
```

But tuples with different lengths are presumed to be distinct types.

```{code-cell} ipython3
ak.type(ak.Array([
    (1.1, [1]               ),
    (2.2,            "two"  ),
    (3.3, [1, 2, 3], "three"),
]))
```

More control over conversions
-----------------------------

The conversions described above are applied by [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) when it maps data into an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html). For more control over the conversion process (e.g. to make unions of records), use [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) directly.
