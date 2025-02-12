# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ctypes
from abc import abstractmethod
from typing import Any, Callable

import awkward as ak
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.typetracer import try_touch_data
from awkward._nplikes.virtual import materialize_if_virtual
from awkward._typing import Protocol, TypeAlias

KernelKeyType: TypeAlias = tuple  # Tuple[str, Unpack[Tuple[metadata.dtype, ...]]]


numpy = Numpy.instance()
metadata = NumpyMetadata.instance()


class KernelError(Protocol):
    filename: bytes | None
    str: bytes | None
    attempt: int
    id: int


class Kernel(Protocol):
    @property
    @abstractmethod
    def key(self) -> KernelKeyType: ...

    @abstractmethod
    def __call__(self, *args) -> KernelError | None:
        raise NotImplementedError


class BaseKernel(Kernel):
    _impl: Callable[..., Any]
    _key: KernelKeyType

    def __init__(self, impl: Callable[..., Any], key: KernelKeyType):
        self._impl = impl
        self._key = key

    @property
    def key(self) -> KernelKeyType:
        return self._key

    def __repr__(self):
        return "<{} {}{}>".format(
            type(self).__name__,
            self.key[0],
            "".join(", " + str(metadata.dtype(x)) for x in self.key[1:]),
        )


class CTypesFunc(Protocol):
    argtypes: tuple[Any, ...]

    def __call__(self, *args) -> Any: ...


class NumpyKernel(BaseKernel):
    @classmethod
    def _cast(cls, x, t):
        if issubclass(t, ctypes._Pointer):
            # Do we have a NumPy-owned array?
            if numpy.is_own_array(x):
                assert numpy.is_c_contiguous(x), "kernel expects contiguous array"
                if x.ndim > 0:
                    return ctypes.cast(x.ctypes.data, t)
                else:
                    return x
            # Or, do we have a ctypes type
            elif hasattr(x, "_b_base_"):
                return ctypes.cast(x, t)
            else:
                raise AssertionError(
                    f"Only NumPy buffers should be passed to Numpy Kernels, received {type(t).__name__}"
                )
        else:
            return x

    def __call__(self, *args) -> None:
        assert len(args) == len(self._impl.argtypes)

        args = materialize_if_virtual(*args)

        return self._impl(
            *(self._cast(x, t) for x, t in zip(args, self._impl.argtypes))
        )


class JaxKernel(NumpyKernel):
    def __call__(self, *args) -> None:
        assert len(args) == len(self._impl.argtypes)

        if not any(Jax.is_tracer_type(type(arg)) for arg in args):
            return super().__call__(*args)


class CupyKernel(BaseKernel):
    def __init__(self, impl: Callable[..., Any], key: KernelKeyType):
        super().__init__(impl, key)

        self._cupy = Cupy.instance()

    def max_length(self, args):
        max_length = metadata.iinfo(metadata.int64).min
        # TODO should kernels strip nplike wrapper? Probably
        for array in args:
            if self._cupy.is_own_array(array):
                max_length = max(max_length, array.size)
        return max_length

    def calc_grid(self, length):
        # CUDA blocks are limited to 1024 threads per block, so to
        # have more than one block, we have at least `length // 1024` blocks
        # of size 1024.
        return (length // 1024) + 1, 1, 1

    def calc_blocks(self, length):
        # CUDA blocks are limited to 1024 threads per block
        # Number of threads are given by `length`
        return min(length, 1024), 1, 1

    def _cast(self, x, type_):
        if type_:
            # Do we have a CuPy-owned array?
            if self._cupy.is_own_array(x):
                assert self._cupy.is_c_contiguous(x)
            return x
        else:
            return x

    def __call__(self, *args) -> None:
        import awkward._connect.cuda as ak_cuda

        args = materialize_if_virtual(*args)

        cupy = ak_cuda.import_cupy("Awkward Arrays with CUDA")
        maxlength = self.max_length(args)
        grid, blocks = self.calc_grid(maxlength), self.calc_blocks(maxlength)
        cupy_stream_ptr = cupy.cuda.get_current_stream().ptr

        if cupy_stream_ptr not in ak_cuda.cuda_streamptr_to_contexts:
            ak_cuda.cuda_streamptr_to_contexts[cupy_stream_ptr] = (
                cupy.array(ak_cuda.NO_ERROR),
                [],
            )
        assert len(args) == len(self._impl.is_ptr)

        args = [self._cast(x, t) for x, t in zip(args, self._impl.is_ptr)]

        # The first arg is the invocation index which raises itself by 8 in the kernel if there was no error before.
        # The second arg is the error_code.
        args = (
            *args,
            len(ak_cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][1]),
            ak_cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][0],
        )
        ak_cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][1].append(
            ak_cuda.Invocation(
                name=self.key[0],
                error_context=ak._errors.ErrorContext.primary(),
            )
        )

        self._impl(grid, blocks, args)


class TypeTracerKernelError(KernelError):
    def __init__(self):
        self.str = None
        self.filename = None
        self.attempt = ak._util.kSliceNone
        self.id = ak._util.kSliceNone


class TypeTracerKernel:
    def __init__(self, index):
        self._name_and_types = index

    def __call__(self, *args) -> TypeTracerKernelError:
        for arg in args:
            try_touch_data(arg)
        return TypeTracerKernelError()

    def __repr__(self):
        return "<{} {}{}>".format(
            type(self).__name__,
            self._name_and_types[0],
            "".join(", " + str(metadata.dtype(x)) for x in self._name_and_types[1:]),
        )
