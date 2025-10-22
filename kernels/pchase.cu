#include <cudaHelpers.cuh>
#include <pchase.cuh>

__global__ void pchaseKernel(uint64_t* array, uint64_t iters, uint64_t* cycles) {
  uint64_t j = 0;
  uint64_t start, end;

  /* Warm the cache. */
  for (auto i = 0; i < iters; i++) /* TODO: warmup takes as long as experiment. */
    j = array[j];

  start = clock64();
  for (uint64_t i = 0; i < iters; i++)
    j = array[j];
  end = clock64();
  *cycles = end - start;

  /* Prevent compiler optimization. */
  if (j == UINT64_MAX)
    array[0] = j;
}

void launchPChaseKernel(uint64_t* array, uint64_t arraySize, uint64_t iters, uint64_t* total_cycles) {
  cudaError_t err;
  uint64_t* deviceArr;
  uint64_t* deviceCycles;

  err = cudaMalloc(&deviceArr, arraySize);
  throwOnErr(err);

  err = cudaMalloc(&deviceCycles, sizeof(uint64_t));
  throwOnErr(err);

  err = cudaMemcpy(deviceArr, array, arraySize, cudaMemcpyHostToDevice);
  throwOnErr(err);

  /* We want to serialize accesses for memory latency measurements, so run with exactly one thread. */
  pchaseKernel<<<1, 1>>>(deviceArr, iters, total_cycles);

  err = cudaDeviceSynchronize();
  throwOnErr(err);

  err = cudaGetLastError();
  throwOnErr(err);

  err = cudaMemcpy(total_cycles, deviceCycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  throwOnErr(err);

  err = cudaFree(deviceArr);
  throwOnErr(err);
}
