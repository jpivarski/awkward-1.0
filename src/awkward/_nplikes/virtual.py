from __future__ import annotations

import typing as tp
import math
from awkward._nplikes.numpy_like import NumpyLike
from awkward._typing import DType
from awkward._nplikes.shape import ShapeItem


class VirtualLeafArrayProxy:
    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DType,
        generator: tp.Callable,
        form_key: str | None = None,
    ) -> None:
        self._nplike = nplike
        self._shape = shape
        self._dtype = dtype
        self._array = None
        self._generator = generator
        self._form_key = form_key

    @property
    def nplike(self) -> NumpyLike:
        return self._nplike

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def generator(self) -> tp.Callable:
        return self._generator

    @property
    def form_key(self) -> str | None:
        return self._form_key

    def materialize(self):
        if self._array is None:
            print("Materializing:", self.form_key)
            self._array = self.nplike.asarray(self.generator())
        return self._array

    @property
    def is_materialized(self) -> bool:
        return self._array is not None

    size = property(lambda self: math.prod(self.shape))
    ndim = property(lambda self: len(self.shape))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def strides(self):
        # Is this safe to assume?
        strides: tuple[ShapeItem, ...] = (self.dtype.itemsize,)
        for item in self.shape[-1:0:-1]:
            strides = (item * strides[0], *strides)
        return strides

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError as e:
            raise TypeError("len() of unsized object") from e  # same as numpy error

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape} dtype={self.dtype}, materialized={self.is_materialized})"
