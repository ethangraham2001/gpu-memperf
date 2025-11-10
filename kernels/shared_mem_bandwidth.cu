#include <cuda_runtime.h>
#include "cudaHelpers.cuh"

__global__ void sharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t stride, uint32_t mode) {
  extern __shared__ uint32_t sharedMem[];
  const uint32_t tid = threadIdx.x;
  uint32_t tmp = 0u;
  const uint32_t mask = numElems - 1u;

  /* Initialize shared memory to avoid compiler optimizations and broadcasts. */
  for (uint32_t idx = tid; idx < numElems; idx += blockDim.x) {
    sharedMem[idx] = idx;
  }
  __syncthreads();

  /* Loop over shared memory with given stride. */
  for (uint32_t i = 0; i < numIters; ++i) {
    uint32_t offset = ((tid * stride) + i) & mask;
    uint32_t val = sharedMem[offset];

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

  /* Prevent compiler optimiziation. */
  if (tmp == 0xFFFFFFFFu) {
    sharedMem[0] = tmp;
  }
  __syncthreads();
}

void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                    uint32_t stride, uint32_t mode, float* elapsedMsOut) {
  cudaError_t err;

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  throwOnErr(err);
  err = cudaEventCreate(&stop);
  throwOnErr(err);

  const dim3 block(threads);

  /* Warmup launch. */
  const uint32_t warmupIters_ = 256;
  sharedMemBandwidthKernel<<<1, block, sharedBytes>>>(numElems, warmupIters_, stride);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  /* Record start event. */
  err = cudaEventRecord(start);
  throwOnErr(err);

  /* Kernel uses extern shared memory size = sharedBytes. */
  sharedMemBandwidthKernel<<<1, block, sharedBytes>>>(numElems, numIters, stride);

  /* Record stop event. */
  err = cudaEventRecord(stop);
  throwOnErr(err);

  err = cudaEventSynchronize(stop);
  throwOnErr(err);

  float ms = 0.0f;
  err = cudaEventElapsedTime(&ms, start, stop);
  throwOnErr(err);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  err = cudaGetLastError();
  throwOnErr(err);

  if (elapsedMsOut)
    *elapsedMsOut = ms;
}
