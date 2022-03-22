// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename C, typename T>
__global__ void
awkward_ListOffsetArray_compact_offsets(T* tooffsets,
                                        const C* fromoffsets,
                                        int64_t length,
                                        uint64_t invocation_index,
                                        uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t diff = (int64_t)fromoffsets[0];
    if (thread_id == 0) {
      tooffsets[0] = 0;
    }
    tooffsets[thread_id + 1] = fromoffsets[thread_id + 1] - diff;
  }
}
