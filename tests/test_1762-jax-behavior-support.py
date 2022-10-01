# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()


def test_behavior():
    class GradBehavior(ak.Array):
        ...

    def square_fifth_entry(x):
        return x[4] ** 2

    primal = ak.Array(np.arange(10, dtype=np.float64), backend="jax")
    tangent = ak.Array(np.arange(10, dtype=np.float64), backend="jax")

    behavior = {"grad": GradBehavior}
    primal_grad = ak.with_parameter(primal, "__array__", "grad", behavior=behavior)
    tangent_grad = ak.with_parameter(tangent, "__array__", "grad", behavior=behavior)
    value_jvp_grad, jvp_grad_grad = jax.jvp(
        square_fifth_entry, (primal_grad,), (tangent_grad,)
    )

    assert value_jvp_grad == pytest.approx(16.0)
    assert jvp_grad_grad == pytest.approx(32.0)


def test_jvp_nested_list():
    # with jax.checking_leaks():
    array = ak.Array(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), backend="jax")
    tangent = ak.Array(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]), backend="jax")

    def func(x):
        return x[::-1] ** 2

    with jax.checking_leaks():
        value_jvp, jvp_grad = jax.jvp(func, (array,), (tangent,))
        assert value_jvp.to_list() == [[1.0, 4.0, 9.0, 16.0, 25.0]]
        assert jvp_grad.to_list() == [[0.0, 0.0, 0.0, 0.0, 10.0]]


def test_recursively_apply_trim():
    numpy = ak.nplikes.Numpy.instance()

    layout = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(numpy.arange(24)),
    )

    # Test trimmed
    visitor_was_called = False

    def visitor(layout, depth, **kwargs):
        nonlocal visitor_was_called
        visitor_was_called = True

        if depth == 1:
            assert len(layout) == 2
        elif depth == 2:
            assert len(layout) == 6

    layout.recursively_apply(visitor, trim=True)
    assert visitor_was_called


def test_recursively_apply_no_trim():
    numpy = ak.nplikes.Numpy.instance()

    layout = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(numpy.arange(24)),
    )

    # Test untrimmed
    visitor_was_called = False

    def visitor(layout, depth, **kwargs):
        nonlocal visitor_was_called
        visitor_was_called = True

        if depth == 1:
            assert len(layout) == 2
        elif depth == 2:
            assert len(layout) == 24

    layout.recursively_apply(visitor, trim=False)
    assert visitor_was_called


def test_ak_transform_trim():
    numpy = ak.nplikes.Numpy.instance()

    layout = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(numpy.arange(24)),
    )

    # Test trimmed
    visitor_was_called = False

    def visitor(layout, depth, **kwargs):
        nonlocal visitor_was_called
        visitor_was_called = True

        if depth == 1:
            assert len(layout) == 2
        elif depth == 2:
            assert len(layout) == 6

    ak.transform(visitor, layout, trim=True)
    assert visitor_was_called


def test_ak_transform_no_trim():
    numpy = ak.nplikes.Numpy.instance()

    layout = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(numpy.arange(24)),
    )

    # Test untrimmed
    visitor_was_called = False

    def visitor(layout, depth, **kwargs):
        nonlocal visitor_was_called
        visitor_was_called = True

        if depth == 1:
            assert len(layout) == 2
        elif depth == 2:
            assert len(layout) == 24

    ak.transform(visitor, layout, trim=False)
    assert visitor_was_called
