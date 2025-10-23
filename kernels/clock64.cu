#include <cuda.h>

#include <cudaHelpers.cuh>

__global__ void clock64OverheadKernel(uint64_t iters, uint64_t* cycles) {
  uint64_t total = 0;

  /* Warmup - clock64() should be in instruction cache. For the benchmark */
  for (uint64_t i = 0; i < iters; i++) {
    uint64_t start = clock64();
    uint64_t end = clock64();
    total += (end - start);
  }
  /* Prevent optimization. */
  if (total == UINT64_MAX) {
    *cycles = total;
    return;
  }

  total = 0;
  for (uint64_t i = 0; i < iters; i++) {
    uint64_t start = clock64();
    uint64_t end = clock64();
    total += (end - start);
  }
  *cycles = total;
}

double measureClock64Latency(uint64_t iters) {
  cudaError_t err;
  uint64_t* dCycles;
  uint64_t hCycles;

  err = cudaMalloc(&dCycles, sizeof(uint64_t));
  throwOnErr(err);

  clock64OverheadKernel<<<1, 1>>>(iters, dCycles);

  err = cudaDeviceSynchronize();
  throwOnErr(err);

  err = cudaGetLastError();
  throwOnErr(err);

  err = cudaMemcpy(&hCycles, dCycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  throwOnErr(err);

  return (double)hCycles / ((double)iters * 2.0);
}
