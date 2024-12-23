# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.typetracer import TypeTracer
from awkward.typetracer import typetracer_with_report

typetracer = TypeTracer.instance()


def test_no_copy():
    form = ak.forms.NumpyForm("int64", form_key="buffer")
    layout, report = typetracer_with_report(form)

    typetracer.asarray(layout.data)
    assert not report.data_touched


def test_no_copy_dtype():
    form = ak.forms.NumpyForm("int64", form_key="buffer")
    layout, report = typetracer_with_report(form)

    typetracer.asarray(layout.data, dtype=np.int64)
    assert not report.data_touched


def test_copy_touch():
    form = ak.forms.NumpyForm("int64", form_key="buffer")
    layout, report = typetracer_with_report(form)

    typetracer.asarray(layout.data, dtype=np.int32)
    assert report.data_touched == ["buffer"]
