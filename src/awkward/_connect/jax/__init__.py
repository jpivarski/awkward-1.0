import jax.numpy

from awkward._connect.jax.reducers import get_jax_reducer  # noqa: F401
from awkward._connect.jax.trees import jax_flatten, jax_unflatten  # noqa: F401


def get_jax_ufunc(ufunc):
    return getattr(jax.numpy, ufunc.__name__)
