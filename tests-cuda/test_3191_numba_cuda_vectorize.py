from __future__ import annotations

import cupy as cp
import numba as nb

import awkward as ak

ak.numba.register_and_check()


@nb.vectorize()
def _square(x):
    return x * x


@nb.vectorize(
    target="cuda",
)
def _square_cuda(x):
    return x * x


counts = cp.random.poisson(lam=3, size=50)
flat_values = cp.random.normal(size=int(counts.sum()))
values = ak.unflatten(flat_values, counts)


def test_square():
    values2_cpu = _square(ak.to_backend(values, "cpu"))
    print(values2_cpu)


def test_square_cuda():
    values2 = _square_cuda(values)
    print(values2)
