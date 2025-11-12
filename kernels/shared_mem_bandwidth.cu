#include <cuda_runtime.h>
#include <clock64.hh>
#include <cudaHelpers.cuh>

__global__ void sharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t stride, uint32_t mode,
                                         uint64_t* cycle) {
  extern __shared__ uint32_t sharedMem[];
  const uint32_t tid = threadIdx.x;
  uint32_t tmp = 0u;
  const uint32_t mask = numElems - 1u;

  /* Initialize shared memory to avoid compiler optimizations and broadcasts. */
  for (uint32_t idx = tid; idx < numElems; idx += blockDim.x) {
    sharedMem[idx] = idx;
  }
  __syncthreads();

  uint64_t start = clock64();

  /* Loop over shared memory with given stride. */
  for (uint32_t i = 0; i < numIters; ++i) {
    uint32_t offset = ((tid * stride) + i) & mask;
    uint32_t dst = (offset + (blockDim.x >> 1)) & mask;

    if (mode == 0) {
      /* Read-only. */
      tmp += sharedMem[offset];
    } else if (mode == 1) {
      /* Write-only. */
      sharedMem[dst] = tid + i;
    } else {
      /* Read + write. */
      uint32_t val = sharedMem[offset];
      sharedMem[dst] = val + 1u;
      tmp += val;
    }
  }

  uint64_t end = clock64();

  /* Prevent compiler optimiziation. */
  if (tmp == 0xFFFFFFFFu) {
    sharedMem[0] = tmp;
  }
  __syncthreads();

  *cycle = end - start;
}

void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                    uint32_t stride, uint32_t mode, uint64_t* cycle) {
  cudaError_t err;
  const dim3 block(threads);

  /* Warmup launch. */
  uint64_t warmupCycle = 0;
  const uint32_t warmupIters_ = 256;
  sharedMemBandwidthKernel<<<1, block, sharedBytes>>>(numElems, warmupIters_, stride, mode, &warmupCycle);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  /* Kernel uses extern shared memory size = sharedBytes. */
  sharedMemBandwidthKernel<<<1, block, sharedBytes>>>(numElems, numIters, stride, mode, cycle);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  err = cudaGetLastError();
  throwOnErr(err);
}
