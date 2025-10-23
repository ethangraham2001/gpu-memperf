#ifndef CUDAHELPERS_CUH
#define CUDAHELPERS_CUH

#include <cuda.h>
#include <stdexcept>

static void throwOnErr(cudaError_t err) {
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
}

#endif /* CUDAHELPERS_CUH */
