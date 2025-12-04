#ifndef CUDAHELPERS_CUH
#define CUDAHELPERS_CUH

#include <cuda.h>
#include <stdexcept>

#define __used __attribute__((used))

#define throwOnErr(err)                                                                                        \
  do {                                                                                                         \
    cudaError_t e = err;                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + " - " + \
                               cudaGetErrorString(e));                                                         \
    }                                                                                                          \
  } while (0)

/**
 * n % x == n & (x - 1) when x is a power of 2.
 *
 * A modulo operator is expensive in general, especially when we are trying to
 * benchmark raw memory performance.
 */
#define mod_power_of_2(x) &(x - 1)

#define isPowerOf2(x) (((x) > 0) && (((x) & ((x) - 1)) == 0))

#endif /* CUDAHELPERS_CUH */
