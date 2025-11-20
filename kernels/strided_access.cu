#include <cuda.h>
#include <CachePolicy.hh>
#include <cudaHelpers.cuh>
#include <vector>
#include "cache_loaders.cuh"

/**
 * stridedReadKernel - Measures memory bandwidth (L1, L2, or DRAM) for strided accesses.
 *
 * Each thread performs a series of strided global loads from @data. Depending on
 * the total data size and stride, the working set may reside in L1, L2, or DRAM,
 * allowing bandwidth characterization at different hierarchy levels.
 *
 * For the i-th access by a thread with global ID tid = blockIdx.x * blockDim.x + threadIdx.x,
 * the accessed index is computed as:
 *
 *     idx = ((tid + i) * stride) & (numElems - 1)
 *
 * Parameters:
 * @data         Pointer to the device array containing elements of any POD type
 *               (e.g., uint32_t, float, double). Template or type-agnostic usage supported.
 * @numElems     Number of elements in @data. Must be a power of two to enable
 *               efficient modular indexing.
 * @stride       Access stride, in elements, between successive loads.
 * @iters        Number of strided loads performed per thread during both the
 *               warm-up and the timed phase.
 * @sink         Per-thread accumulator to ensure that memory loads are not optimized away.
 * @totalCycles  Elapsed cycles for the timed phase, recorded by thread 0 of each block.
 *               In multi-block launches, the last writer overwrites this value
 *               unless a per-block buffer is used.
 */

template <typename T, typename Loader>
__global__ void stridedReadKernel(const T* data, uint64_t numElems, uint64_t stride, uint64_t iters, uint64_t* sink,
                                  uint64_t* blockCycles, Loader load) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t totalThreads = gridDim.x * blockDim.x;

  __shared__ uint64_t start, end;

  /* Accumulate to prevent the compiler from eliding loads. */
  uint64_t acc = 0;

  /* Warmup to populate L1. */
  for (uint64_t i = 0; i < iters; i++) {
    uint64_t idx = ((tid + i * totalThreads) * stride) mod_power_of_2(numElems);
    load(&data[idx], acc);
  }

  __syncthreads();

  if (threadIdx.x == 0)
    start = clock64();

  __syncthreads();

  /* Timed section: perform iters strided loads. */
  for (uint64_t i = 0; i < iters; i++) {
    uint64_t idx = ((tid + i * totalThreads) * stride) mod_power_of_2(numElems);
    load(&data[idx], acc);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    end = clock64();
    blockCycles[blockIdx.x] = end - start;
  }

  sink[tid] = acc;
}

template <typename T>
void launchStridedAccessKernel(const std::vector<T>& data, cacheload::CachePolicy policy, uint64_t stride,
                               uint64_t iters, uint64_t threadsPerBlock, uint64_t numBlocks, uint64_t* totalCycles) {

  uint64_t totalThreads = threadsPerBlock * numBlocks;

  T* dData;
  uint64_t* dSink;
  uint64_t* dBlockCycles;

  throwOnErr(cudaMalloc(&dData, data.size() * sizeof(T)));
  throwOnErr(cudaMemcpy(dData, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dSink, totalThreads * sizeof(uint64_t)));
  throwOnErr(cudaMalloc(&dBlockCycles, numBlocks * sizeof(uint64_t)));

  switch (policy) {
    case cacheload::CachePolicy::L1: {
      stridedReadKernel<T, cacheload::L1Loader>
          <<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
              dData, data.size(), stride, iters, dSink, dBlockCycles, cacheload::L1Loader{});
      break;
    }
    case cacheload::CachePolicy::L2: {
      stridedReadKernel<T, cacheload::L2Loader>
          <<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
              dData, data.size(), stride, iters, dSink, dBlockCycles, cacheload::L2Loader{});
      break;
    }
    case cacheload::CachePolicy::DRAM: {
      stridedReadKernel<T, cacheload::DramLoader>
          <<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
              dData, data.size(), stride, iters, dSink, dBlockCycles, cacheload::DramLoader{});
      break;
    }
  }

  throwOnErr(cudaDeviceSynchronize());

  throwOnErr(cudaGetLastError());

  std::vector<uint64_t> hCycles(numBlocks);
  throwOnErr(cudaMemcpy(hCycles.data(), dBlockCycles, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  uint64_t maxCycles = 0;
  for (auto c : hCycles)
    if (c > maxCycles)
      maxCycles = c;
  if (totalCycles)
    *totalCycles = maxCycles;

  cudaFree(dData);
  cudaFree(dSink);
  cudaFree(dBlockCycles);
}

/* The compiler complains when the concrete versions that we use aren't defined.
 * The implementation isn't required - just a header. So we declare a concrete
 * header for each version of stridedAccessKernel that we use. */

template void launchStridedAccessKernel<types::f8>(const std::vector<types::f8>&, cacheload::CachePolicy, uint64_t,
                                                   uint64_t, uint64_t, uint64_t, uint64_t*);

template void launchStridedAccessKernel<types::f16>(const std::vector<types::f16>&, cacheload::CachePolicy, uint64_t,
                                                    uint64_t, uint64_t, uint64_t, uint64_t*);

template void launchStridedAccessKernel<types::f32>(const std::vector<types::f32>&, cacheload::CachePolicy, uint64_t,
                                                    uint64_t, uint64_t, uint64_t, uint64_t*);

template void launchStridedAccessKernel<types::f64>(const std::vector<types::f64>&, cacheload::CachePolicy, uint64_t,
                                                    uint64_t, uint64_t, uint64_t, uint64_t*);
