// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax(T* toptr,
                      const C* fromptr,
                      const U* parents,
                      int64_t lenparents,
                      int64_t outlength,
                      uint64_t invocation_index,
                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = -1;
    }
    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      if (toptr[parent] == -1 ||
          (fromptr[thread_id] > (fromptr[toptr[parent]]))) {
        toptr[parent] = thread_id;
      }
    }
  }
}
