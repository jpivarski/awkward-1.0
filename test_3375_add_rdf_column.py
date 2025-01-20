# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare

import numpy
import awkward as ak
from ROOT import RDF, RDataFrame

# ---------------------------------------------------------------------
def add_column(rdf : RDataFrame, arr_val : numpy.ndarray, name : str):
    v_col_org = rdf.GetColumnNames()
    l_col_org = [name.c_str() for name in v_col_org ]
    l_col     = []

    for col in l_col_org:
        l_col.append(col)

    data  = ak.from_rdataframe(rdf, columns=l_col)
    d_data= { col : data[col] for col in l_col }

    d_data[name] = arr_val

    rdf = ak.to_rdataframe(d_data)

    return rdf

# ---------------------------------------------------------------------
def test_add_column():
    ak_array_x = ak.Array([1, 2, 3])
    ak_array_y = ak.Array([4, 5, 6])

    rdf = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})

    rdf = rdf.Define('z', 'ROOT::RVec<int>({1, 2, 3})')
    rdf = rdf.Define('w', 'true')

    arr_val = ak.Array([10, 20, 30])

    rdf = add_column(rdf, arr_val, 'values')

    rdf.Display().Print()

# ---------------------------------------------------------------------
def add_numpy_column(rdf : RDataFrame, arr_val : numpy.ndarray, name : str):
    v_col_org = rdf.GetColumnNames()
    l_col_org = [name.c_str() for name in v_col_org ]
    l_col     = []

    for col in l_col_org:
        l_col.append(col)

    data  = ak.from_rdataframe(rdf, columns=l_col)
    d_data= { col : data[col] for col in l_col }

    d_data[name] = arr_val

    rdf = ak.to_rdataframe(d_data)

    return rdf
    
# ---------------------------------------------------------------------
def test_add_numpy():
    d_data = {
            'x' : numpy.array([1, 2, 3]),
            'y' : numpy.array([4, 5, 6]),
            }

    rdf = RDF.MakeNumpyDataFrame(d_data)
    rdf = rdf.Define('z', 'ROOT::RVec<int>({1, 2, 3})')
    rdf = rdf.Define('w', 'true')

    arr_val = numpy.array([10, 20, 30])

    rdf = add_numpy_column(rdf, arr_val, 'values')

    rdf.Display().Print()
