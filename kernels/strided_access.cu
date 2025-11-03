#include <cuda.h>
#include <cudaHelpers.cuh>
#include <vector>

/**
 * stridedReadKernel - L1 strided access bandwidth measurement kernel
 *
 * Each thread performs strided global loads from @data. For the i-th access of
 * a thread with global id tid = blockIdx.x * blockDim.x + threadIdx.x, the
 * index is: idx = ((tid + i) * stride) & (numElems - 1)
 *
 * @data: device pointer to uint32_t data array (TODO: support other types)
 * @numElems: number of elements in @data (Should be a power of two); 
 * @stride: access stride in elements between successive loads
 * @iters: number of strided loads per thread in warmup and in the timed phase
 * @sink: per-thread accumulator to keep loads observable
 * @totalCycles: elapsed cycles for the timed phase as measured by thread 0
 *                  of each block; for multi-block launches, the last writer wins
 *                  unless a per-block buffer is used
 */
__global__ void stridedReadKernel(const uint32_t* data, uint64_t numElems, uint64_t stride,
                                  uint64_t iters, uint64_t* sink, uint64_t* totalCycles) {

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t sharedStart, sharedEnd;

  /* Accumulate to prevent the compiler from eliding loads. */
  uint64_t acc = 0;

  /* Warmup to populate L1. */
  for (uint64_t i = 0; i < iters ; i++) {
    uint64_t idx = ((tid + i) * stride) mod_power_of_2(numElems);
    acc += static_cast<uint64_t>(data[idx]);
  }

  __syncthreads();

  if (threadIdx.x == 0)
    sharedStart = clock64();

  __syncthreads();

  /* Timed section: perform iters strided loads. */
  for (uint64_t i = 0; i < iters; i++) {
    uint64_t idx = ((tid + i) * stride) mod_power_of_2(numElems);
    acc += static_cast<uint64_t>(data[idx]);
  }
  
  __syncthreads();

  if (threadIdx.x == 0) {
    sharedEnd = clock64();
  }

  *totalCycles = sharedEnd - sharedStart;
  sink[tid] = acc;
}

void launchStridedAccessKernel(const std::vector<uint32_t> data, uint64_t stride, uint64_t iters,
                               uint64_t threadsPerBlock, uint64_t numBlocks, uint64_t* totalCycles) {

  uint64_t totalThreads = threadsPerBlock * numBlocks;

  uint32_t* dData;
  uint64_t* dSink;
  uint64_t* dSharedCycles;

  throwOnErr(cudaMalloc(&dData, data.size() * sizeof(uint32_t)));
  throwOnErr(cudaMemcpy(dData, data.data(), data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dSink, totalThreads * sizeof(uint64_t)));
  throwOnErr(cudaMalloc(&dSharedCycles, sizeof(uint64_t)));

  stridedReadKernel<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(dData, data.size(), stride, iters, dSink, dSharedCycles);
  
  throwOnErr(cudaDeviceSynchronize());

  throwOnErr(cudaGetLastError());
  
  throwOnErr(cudaMemcpy(totalCycles, dSharedCycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  cudaFree(dData);
  cudaFree(dSink);
  cudaFree(dSharedCycles);
}
