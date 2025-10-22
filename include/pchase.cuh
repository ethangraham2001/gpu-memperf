#ifndef PCHASE_CUH
#define PCHASE_CUH

#include <cuda.h>

/**
 * pchaseKernel - a cuda kernel for measuring average memory access latency
 */
__global__ void pchaseKernel(uint64_t* array, uint64_t iters, uint64_t* total_cycles);

/**
 * launchPChaseKernel - host-callable wrapper
 */
void launchPChaseKernel(uint64_t* array, uint64_t arraySize, uint64_t iters, uint64_t* total_cycles);

#endif /* PCHASE_CUH */
