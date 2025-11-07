#include <cuda_runtime.h>
#include "cudaHelpers.cuh"

__global__ void sharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t stride) {
  extern __shared__ uint32_t sharedMem[];
  const uint32_t tid = threadIdx.x;
  uint32_t tmp = 0u;
  const uint32_t mask = numElems - 1u;

  /* Initialize shared memory to avoid compiler optimizations and broadcasts. */
  for (uint32_t idx = tid; idx < numElems; idx += blockDim.x) {
    sharedMem[idx] = idx;
  }
  __syncthreads();

  /* Measured loop performs one read and one write with given stride. */
  for (uint32_t i = 0; i < numIters; ++i) {
    uint32_t offset = ((tid * stride) + i) & mask;
    uint32_t val = sharedMem[offset];
    uint32_t dst = (offset + (blockDim.x >> 1)) & mask;
    sharedMem[dst] = val + 1u;
    tmp += val;
  }

  /* Prevent compiler optimiziation. */
  if (tmp == 0xFFFFFFFFu) {
    sharedMem[0] = tmp;
  }
  __syncthreads();
}

void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                    uint32_t stride, float* elapsedMsOut) {
  cudaError_t err;

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  throwOnErr(err);
  err = cudaEventCreate(&stop);
  throwOnErr(err);

  const dim3 grid(1);
  const dim3 block(threads);

  /* Warmup launch. */
  sharedMemBandwidthKernel<<<grid, block, sharedBytes>>>(numElems, 256, stride);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  /* Record start event. */
  err = cudaEventRecord(start);
  throwOnErr(err);

  /* Kernel uses extern shared memory size = sharedBytes. */
  sharedMemBandwidthKernel<<<grid, block, sharedBytes>>>(numElems, numIters, stride);

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
