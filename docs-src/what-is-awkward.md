---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.4.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

What is an Awkward Array?
=========================

Efficiency and generality
-------------------------

Arrays are the most efficient data structures in computing, and [NumPy](https://numpy.org/) makes it easy to interact with arrays in Python. However, NumPy's arrays are rectangular tables or tensors that cannot express variable-length structures.

General tree-like data are often expressed using [JSON](https://www.json.org/), but at the expense of memory use and processing speed.

Awkward Arrays are general tree-like data structures, like JSON, but contiguous in memory and operated upon with compiled, vectorized code like NumPy. They're building blocks for data analysis, recognizing that some datasets are, well, more awkward than others.

This library was originally developed for high-energy particle physics. Particle physics datasets have rich data structures that usually can't be flattened into rectangular arrays, but physicists need to process them efficiently because the datasets are enormous. Awkward Arrays combine generic data structures with high-performance number-crunching.

Let's illustrate this with a non-physics dataset: maps of bike routes in my hometown of Chicago.

JSON to array
-------------

The city of Chicago publishes quite a lot of data; here is a [GeoJSON of bike paths](https://github.com/Chicago/osd-bike-routes/blob/master/data/Bikeroutes.geojson):

<div style="margin-right: 15px">
  <iframe class="render-viewer " src="https://render.githubusercontent.com/view/geojson?commit=5f556dcd4ed54f5f5c926c01c34ebc6261ec7d34&amp;enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4368696361676f2f6f73642d62696b652d726f757465732f356635353664636434656435346635663563393236633031633334656263363236316563376433342f646174612f42696b65726f757465732e67656f6a736f6e&amp;nwo=Chicago%2Fosd-bike-routes&amp;path=data%2FBikeroutes.geojson&amp;repository_id=8065965&amp;repository_type=Repository#292b99b1-4187-4d13-8b0d-ce8fa6eb4ab1" sandbox="allow-scripts allow-same-origin allow-top-navigation" title="File display" width="100%" height="400px"></iframe>
</div>

which we load into Python by downloading it from GitHub and converting it from JSON.

```{code-cell}
import urllib.request
import json

url = "https://raw.githubusercontent.com/Chicago/osd-bike-routes/master/data/Bikeroutes.geojson"
bikeroutes_json = urllib.request.urlopen(url).read()
bikeroutes_pyobj = json.loads(bikeroutes_json)
```

It's a complex dataset with nested structures containing street names and variable-length polylines, and it's about 2 MB.

To load it as an Awkward Array, we can use the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor for an array of records or the [ak.Record](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Record.html) constructor for a single record like this one.

```{code-cell}
import awkward1 as ak
bikeroutes = ak.Record(bikeroutes_pyobj)
bikeroutes
```

Data types
----------

Generic JSON is untyped, but GeoJSON is regular enough that its data type can be inferred. The [ak.type](https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html) function displays an array's type in [Datashape syntax](https://datashape.readthedocs.io/).

```{code-cell}
ak.type(bikeroutes)
```

In the above, `{"field name": type, ...}` denote record structures and `var` indicates variable-length lists. The `"coordinates"` are `var * var * var * float64`, lists of lists of lists of numbers.

Slicing
-------

[NumPy-like slicing](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html#ak-array-getitem) extracts structures within the array. The slice may consist of integers, ranges, and many other slice types, like NumPy, and commas indicate different slices applied to different dimensions. Since our data contain records, we can use strings to select nested records by field name.

```{code-cell}
(bikeroutes["features", "properties", "STREET", 100],
 bikeroutes["features", "geometry", "coordinates", 100, 0])
```

Bike route `100` is on `W DEVON AVE`; it has 7 longitude-latitude coordinates. Here's another:

```{code-cell}
(bikeroutes["features", "properties", "STREET", -1],
 bikeroutes["features", "geometry", "coordinates", -1, 0])
```

The last bike route in the dataset (at index `-1`) is on `N ELSTON AVE` and has 11 coordinates. The type of this array is `11 * var * float64` meaning that the length of each inner list _could be_ variable, though longitude-latitude coordinates ought to all have exactly two elements.

Variable-length lists
---------------------

The [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html) function tells us how many elements are in each list. Applying it to the last bike route, we see that each list indeed has two elements.

```{code-cell}
ak.num(bikeroutes["features", "geometry", "coordinates", -1, 0])
```

This fact was not expressed in the data type because JSON lists don't declare their lengths. In this dataset, they _happened to have_ length 2, but in principle, they could have had different lengths.

The number of coordinate points in each route, on the other hand, is variable:

```{code-cell}
ak.num(bikeroutes["features", "geometry", "coordinates"], axis=2)
```

because a polyline is described by an arbitrary number of longitude-latitude points, while the length of each of those points is 2.

```{code-cell}
ak.num(bikeroutes["features", "geometry", "coordinates"], axis=3)
```

Here, we have used the `axis` parameter to tell [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html) which dimension we're interested in. Most NumPy functions have an `axis` parameter with the same meaning.

We can ask if all coordinates for all streets have length 2 using vectorized `==` and [ak.all](https://awkward-array.readthedocs.io/en/latest/_auto/ak.all.html).

```{code-cell}
ak.all(ak.num(bikeroutes["features", "geometry", "coordinates"], axis=3) == 2)
```

This [ak.all](https://awkward-array.readthedocs.io/en/latest/_auto/ak.all.html) is a strict generalization of [np.all](https://numpy.org/doc/stable/reference/generated/numpy.all.html) in NumPy, and if you have NumPy 1.17 or later, they can be used interchangeably. NumPy's function knows that it needs to defer to ours when it encounters an Awkward Array.

```{code-cell}
import numpy as np
np.all(ak.num(bikeroutes["features", "geometry", "coordinates"], axis=3) == 2)
```

Another curiosity of this dataset is that most of the lists in `axis=1` have length 1,

```{code-cell}
ak.num(bikeroutes["features", "geometry", "coordinates"], axis=1)
```

but not all.

```{code-cell}
np.all(ak.num(bikeroutes["features", "geometry", "coordinates"], axis=1) == 1)
```

Some of these bike routes are discontiguous curves, described by multiple polylines. Just as in NumPy, we can use an [array of booleans as a slice](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html#filtering).

```{code-cell}
selection = ak.num(bikeroutes["features", "geometry", "coordinates"], axis=1) != 1
selection
```

```{code-cell}
ak.num(bikeroutes["features", "geometry", "coordinates"])[selection]
```

They are relatively rare; there are only 11 discontiguous routes, but two of them have over 5 segments! We can apply the same slice to the street names to find out which these are and use [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) to turn the result into a Python list because there are so few of them.

```{code-cell}
ak.to_list(bikeroutes["features", "properties", "STREET"][selection])
```

Array math
----------

Since we now know that the `"coordinates"` are longitude-latitude points, let's convert them into displacements from the center of the map in kilometers. At Chicago's latitude, one degree of longitude is 82.7 km and one degree of latitude is 111.1 km.

Here are a few more tricks: field names can be selected as attributes (with a `.`) and ellipsis (`...`) can be used to pick the last dimension.

```{code-cell}
longitude = bikeroutes.features.geometry.coordinates[..., 0]
latitude = bikeroutes.features.geometry.coordinates[..., 1]
longitude, latitude
```

The [ak.mean](https://awkward-array.readthedocs.io/en/latest/_auto/ak.mean.html) function is equivalent to [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), but it operates on Awkward Arrays. Subtraction or multiplication of arrays [broadcast as in NumPy](https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting), but with [additional rules for Awkward structures](https://awkward-array.readthedocs.io/en/latest/_auto/ak.broadcast_arrays.html).

```{code-cell}
km_east = (longitude - np.mean(longitude)) * 82.7
km_north = (latitude - np.mean(latitude)) * 111.1
km_east, km_north
```

Suppose we want to compute the length of each route. That would be the sum of the lengths of each segment in the route, and a segment is a line joining two consecutive points.

We can drop the last and first point in each polyline with slices like `:-1` and `1:`,

```{code-cell}
km_east[0, 0, :-1], km_east[0, 0, 1:]
```

and thus get the displacement of each segment by subtracting these sliced lists:

```{code-cell}
km_east[0, 0, :-1] - km_east[0, 0, 1:]
```

Doing so for all routes in the entire dataset is one line (using a slice of `:` to mean "get everything").

```{code-cell}
km_east[:, :, :-1] - km_east[:, :, 1:]
```

Using $\sqrt{(x_i - x_{i + 1})^2 + (y_i - y_{i + 1})^2}$ for the distance between points $(x_i, y_i)$ and $(x_{i + 1}, y_{i + 1})$, we can compute all the segment lengths at once

```{code-cell}
segment_length = np.sqrt((km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
                         (km_north[:, :, 1:] - km_north[:, :, :-1])**2)
segment_length
```

and then sum over the variable-length lists of segment lengths to get full route lengths using [ak.sum](https://awkward-array.readthedocs.io/en/latest/_auto/ak.sum.html) (equivalent to [np.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).

```{code-cell}
route_length = np.sum(segment_length, axis=-1)
route_length
```

Notice that `segment_length` has type `1061 * var * var * float64` and `route_length` has type `1061 * var * float64`. It has one fewer `var` dimension because we have summed over it. We can further sum over the discontiguous curves that 11 of the streets have to get total lengths.

```{code-cell}
total_length = np.sum(route_length, axis=-1)
total_length
```

Now there's exactly one of these for each of the 1061 streets.

```{code-cell}
for i in range(10):
    print(bikeroutes.features.properties.STREET[i], "\t\t", total_length[i])
```

This would have been incredibly awkward to write in terms of only NumPy operations, and slow if executed in Python loops.

Performance
-----------

Having well-defined data types and operating on whole arrays in each Python statement allows the bulk of the processing to be performed in precompiled routines. This is the principle behind NumPy as well.

Another factor is that the data are not laid out in memory the way JSON or Python objects are—contiguous by record or scattered about in objects that point to each other—they are laid out as arrays, contiguous by field.

For instance, if we dig down to the coordinate data, we see that it is a NumPy array of numerical values without being interrupted by any character strings from field names or street names.

```{code-cell}
bikeroutes.features.geometry.coordinates.layout.content.content.content
```

This is known as [columnar data](https://towardsdatascience.com/the-beauty-of-column-oriented-data-2945c0c9f560), which has advantages in storage and processing.

Most functions in the Awkward Array library work by unpacking the whole-array structure to find the relevant columns, apply a mathematical function (usually NumPy's own) to the columns, then pack it up in a new structure.

The view of data as nested records is a high-level convenience. Internally, it's all just one-dimensional arrays.

![](img/how-it-works.svg)

To put this in concrete numbers, it means Awkward operations are many times faster than their pure Python equivalents, besides being more concise to express.

**Python for loops:**

```{code-cell}
%%timeit

total_length = []
for route in bikeroutes_pyobj["features"]:
    route_length = []
    for polyline in route["geometry"]["coordinates"]:
        segment_length = []
        last = None
        for lng, lat in polyline:
            km_east = lng * 82.7
            km_north = lat * 111.1
            if last is not None:
                dx2 = (km_east - last[0])**2
                dy2 = (km_north - last[1])**2
                segment_length.append(np.sqrt(dx2 + dy2))
            last = (km_east, km_north)

        route_length.append(sum(segment_length))
    total_length.append(sum(route_length))
```

**Awkward Arrays:**

```{code-cell}
%%timeit

km_east = bikeroutes.features.geometry.coordinates[..., 0] * 82.7
km_north = bikeroutes.features.geometry.coordinates[..., 1] * 111.1

segment_length = np.sqrt((km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
                         (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

route_length = np.sum(segment_length, axis=-1)
total_length = np.sum(route_length, axis=-1)
```

![](img/bikeroutes-scaling.svg)

Parallel processing wasn't included in the above test, but Awkward computations are better for multithreading because compiled routines don't lock the Python interpreter the way that Python statements do (see [Python GIL](https://www.dabeaz.com/python/UnderstandingGIL.pdf)).

Additionally, numerical data take about 8× less memory because they're packed into arrays without per-object overhead.

These are the normal advantages of using array libraries for mathematical calculations over Python objects, but Awkward Array widens the scope of problems that can be solved with arrays.

Compatibility
-------------

Awkward Array is not the only library to target numerical data analysis in Python, but we think it is unique in its emphasis on manipulating columnar data structures.

  * All functions that are defined in both [NumPy](https://numpy.org/) and Awkward Array are strictly generalized in the latter. Rectangular arrays behave the same way in both libraries.
  * [Apache Arrow](https://arrow.apache.org/) is an emerging standard for general columnar data structures like Awkward's, but Arrow is more concerned with sharing arrays between applications than structure manipulation. Most Awkward Arrays can be converted to and from Arrow Arrays without copying the underlying buffers (i.e. conversion is fast and doesn't scale with the size of the dataset). Conversion to Arrow also enables saving as [Parquet files](https://parquet.apache.org/), widely used by columnar databases.
  * [Uproot](https://github.com/scikit-hep/uproot#readme) reads and writes the [ROOT file format](https://root.cern/) as Awkward Arrays. ROOT is ubiquitous in particle physics; more than an exabyte of data is in ROOT format.
  * Awkward Arrays can also be [Pandas](https://pandas.pydata.org/) DataFrame columns without any copying or conversion.
  * Awkward Arrays can also be passed in and out of functions compiled by [Numba](http://numba.pydata.org/), the leading just-in-time compiler for numerical Python. This allows for fast for-loop style calculations on the same data structures.
  * [Dask](https://dask.org/) can perform delayed, distributed, and cache-efficient calculations on Awkward Arrays.
  * [Zarr](https://zarr.readthedocs.io/en/stable/) can store and deliver Awkward Arrays as a v3 extension.

**FIXME:** not all of these are implemented yet, but I didn't want to forget to write them. See [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues) for progress.

Awkward Array "plays well" with the scientific Python ecosystem, enhancing it with one new capability: the ability to analyze JSON-like data with NumPy-like idioms.
