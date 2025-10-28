#include <cuda_runtime.h>
#include "cudaHelpers.cuh"

__global__ void sharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t elemBytes) {
  extern __shared__ uint8_t sharedMemRaw[];
  uint32_t* sharedMem = reinterpret_cast<uint32_t*>(sharedMemRaw);
  const uint32_t tid = threadIdx.x;
  uint32_t tmp = 0u;

  __syncthreads();

  for (uint32_t i = 0; i < numIters; ++i) {
    /* random offset */
    const uint32_t offset = (tid * 37 + i * 17) % numElems;
    /* write */
    sharedMem[offset] = tid + i;
    /* read */
    tmp += sharedMem[offset];
  }

  /* Prevent compiler optimization. */
  if (tmp == 0xFFFFFFFFu)
    sharedMem[0] = tmp;
  __syncthreads();
}

void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, size_t sharedBytes,
                                    float* elapsedMsOut) {
  cudaError_t err;

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  throwOnErr(err);
  err = cudaEventCreate(&stop);
  throwOnErr(err);

  const dim3 grid(1);
  const dim3 block(threads);

  /* Record start event. */
  err = cudaEventRecord(start);
  throwOnErr(err);

  // Kernel uses extern shared memory size = sharedBytes
  sharedMemBandwidthKernel<<<grid, block, sharedBytes>>>(numElems, numIters, 4);

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
