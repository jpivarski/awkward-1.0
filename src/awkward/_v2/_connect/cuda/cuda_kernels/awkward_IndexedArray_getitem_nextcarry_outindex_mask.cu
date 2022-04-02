// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class AWKWARD_INDEXEDARRAY_GETITEM_NEXTCARRY_OUTINDEX_MASK_ERRORS {
  IND_OUT_OF_RANGE // message: "index out of range"
};

// BEGIN PYTHON
// def f(grid, block, tocarry, toindex, fromindex, lenindex, lencontent, invocation_index, err_code):
//     scan_in_array = cupy.empty(lenindex, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_getitem_nextcarry_outindex_mask_a", fromindex.dtype]))(grid, block, (tocarry, toindex, fromindex, scan_in_array, lenindex, lencontent, invocation_index, err_code))
//     scan_in_array = inclusive_scan(scan_in_array, lenindex)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_getitem_nextcarry_outindex_mask_b", fromindex.dtype, tocarry.dtype]))(grid, block, (tocarry, toindex, fromindex, scan_in_array, lenindex, lencontent, invocation_index, err_code))
// out["awkward_IndexedArray_getitem_nextcarry_outindex_mask_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_getitem_nextcarry_outindex_mask_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_IndexedArray_getitem_nextcarry_outindex_mask_a(
  T* tocarry,
  C* toindex,
  const U* fromindex,
  int64_t* scan_in_array,
  int64_t lenindex,
  int64_t lencontent,
  uint64_t* invocation_index,
  uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j >= lencontent) {
        RAISE_ERROR(AWKWARD_INDEXEDARRAY_GETITEM_NEXTCARRY_OUTINDEX_MASK_ERRORS::IND_OUT_OF_RANGE)
      }
      else if (j < 0) {
        scan_in_array[thread_id] = 0;
      }
      else {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__
void awkward_IndexedArray_getitem_nextcarry_outindex_mask_b(
    T* tocarry,
    C* toindex,
    const U* fromindex,
    int64_t* scan_in_array,
    int64_t lenindex,
    int64_t lencontent,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j >= lencontent) {
        RAISE_ERROR(AWKWARD_INDEXEDARRAY_getitem_nextcarry_outindex_mask_ERRORS::IND_OUT_OF_RANGE)
      }
      else if (j < 0) {
        toindex[thread_id] = -1;
      }
      else {
        tocarry[scan_in_array[thread_id] - 1] = j;
        toindex[thread_id] = (C)k;
      }
    }
  }
}
