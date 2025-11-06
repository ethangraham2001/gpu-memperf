#include <vector>

#include <cuda.h>

#include <Types.hh>
#include <cudaHelpers.cuh>
#include <random_access_kernel.hh>

#include <clock64.hh>

/**
 * l1LoadElem - inline PTX for loading an element from L1 cache
 *
 * Constexpr templating means that this if statement is never evaluated at
 * runtime on the device - it is evaluated statically.
 */
template <typename T>
__device__ __forceinline__ void l1LoadElem(T* addr, uint64_t& sink) {
  if constexpr (sizeof(T) == sizeof(types::f8)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.ca.u8 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f16)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.ca.u16 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f32)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.ca.u32 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f64)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.ca.u64 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void l2LoadElem(T* addr, uint64_t& sink) {
  if constexpr (sizeof(T) == sizeof(types::f8)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cg.u8 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f16)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cg.u16 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f32)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cg.u32 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f64)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cg.u64 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  }
}

/**
 * randomAccessKernelL1 - L1 random access bandwidth measurement kernel
 *
 * The host must provide some set of random indices with the same cardinality as
 * the data array which should be a permutation of range [0, numElems - 1].
 *
 * Access to the indices array is perfectly coalesced for maximum efficiency,
 * and the indices contained within it should be well-distributed so that
 * accesses are random.
 *
 * @data: a data array of type T, which whould be a primitive type
 * @indices: a random permutation of range [0, numElems - 1]
 * @numElems: cardinality of @data and @indices
 * @numAccesses: the number of accesses performed per thread
 * @results[numThreads]: number of cycles per thread for the accesses
 * @totalCycles: cycles for all threads to complete as measured by thread 0
 * @sink[numThreads]: used to prevent compiler optimization - can be ignored
 */
template <typename T>
__global__ void randomAccessKernelL1(T* data, uint32_t* indices, uint64_t numElems, uint64_t numAccesses,
                                     uint64_t* results, uint64_t* totalCycles, uint64_t* sink) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t totalThreads = gridDim.x * blockDim.x;
  /* Shared memory to ensure that exactly one copy of each of these exists. */
  __shared__ uint64_t sharedStart, sharedEnd;
  uint64_t localSink = 0;

  /* Warm the cache. */
  for (uint64_t i = 0; i < numAccesses; i++) {
    uint64_t idx = indices[(tid + i * totalThreads) mod_power_of_2(numElems)];
    l1LoadElem(&data[idx], localSink);
  }

  /* Sync on both ends of launching the timer for best accuracy - we don't want
   * any threads to begin accessing data until the timer has started. */
  __syncthreads();
  if (threadIdx.x == 0)
    sharedStart = clock64();
  __syncthreads();

  uint64_t start = clock64();
  for (uint64_t i = 0; i < numAccesses; i++) {
    uint64_t idx = indices[(tid + i * totalThreads) mod_power_of_2(numElems)];
    l1LoadElem(&data[idx], localSink);
  }
  uint64_t end = clock64();

  /* Sync before stopping the global timer. Similarly to above, we don't want to
   * be stopping the timer until all threads have finished their work. */
  __syncthreads();
  if (threadIdx.x == 0) {
    sharedEnd = clock64();
    *totalCycles = sharedEnd - sharedStart;
  }

  results[tid] = end - start;
  sink[tid] = localSink;
}

template <typename T>
__global__ void randomAccessKernelL2(T* data, uint32_t* indices, uint64_t numElems, uint64_t numAccesses,
                                     uint64_t* results, uint64_t* totalCycles, uint64_t* sink) { 
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t totalThreads = gridDim.x * blockDim.x;
  /* Shared memory to ensure that exactly one copy of each of these exists. */
  __shared__ uint64_t sharedStart, sharedEnd;
  uint64_t localSink = 0;

  /* Warm the cache. */
  for (uint64_t i = 0; i < numAccesses; i++) {
  uint64_t idx = indices[(tid + i * totalThreads) mod_power_of_2(numElems)];
  l2LoadElem(&data[idx], localSink);
  }

  /* Sync on both ends of launching the timer for best accuracy - we don't want
  * any threads to begin accessing data until the timer has started. */
  __syncthreads();
  if (threadIdx.x == 0)
  sharedStart = clock64();
  __syncthreads();

  uint64_t start = clock64();
  for (uint64_t i = 0; i < numAccesses; i++) {
  uint64_t idx = indices[(tid + i * totalThreads) mod_power_of_2(numElems)];
  l2LoadElem(&data[idx], localSink);
  }
  uint64_t end = clock64();

  /* Sync before stopping the global timer. Similarly to above, we don't want to
  * be stopping the timer until all threads have finished their work. */
  __syncthreads();
  if (threadIdx.x == 0) {
  sharedEnd = clock64();
  *totalCycles = sharedEnd - sharedStart;
  }

  results[tid] = end - start;
  sink[tid] = localSink;
}

template <typename T>
__global__ void randomAccessKernelDRAM(T* data, uint32_t* indices, uint64_t numElems, uint64_t numAccesses,
                                       uint64_t* results, uint64_t* sharedCycles, uint64_t* sink) { /* TODO */ }

template <typename T>
using randomAccessKernelFunc = void (*)(T*, uint32_t*, uint64_t, uint64_t, uint64_t*, uint64_t*, uint64_t*);

template <typename T>
static randomAccessKernelFunc<T> getKernel(randomAccessKernel::mode mode) {
  switch (mode) {
    case randomAccessKernel::L1_CACHE:
      return randomAccessKernelL1<T>;
    case randomAccessKernel::L2_CACHE:
      return randomAccessKernelL2<T>;
    case randomAccessKernel::DRAM:
      return randomAccessKernelDRAM<T>;
    default:
      throw std::invalid_argument("invalid mode");
  }
}

template <typename T>
std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel(const std::vector<T>& data,
                                                                    const std::vector<uint32_t>& indices,
                                                                    uint64_t numAccesses, uint64_t threadsPerBlock,
                                                                    uint64_t numBlocks, randomAccessKernel::mode mode) {
  uint64_t* dTimingResults;
  uint64_t* dSharedCycles;
  uint32_t* dIndices;
  uint64_t* dSink;
  T* dData;

  uint64_t totalThreads = numBlocks * threadsPerBlock;

  throwOnErr(cudaMalloc(&dData, data.size() * sizeof(T)));
  throwOnErr(cudaMemcpy(dData, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dIndices, indices.size() * sizeof(uint32_t)));
  throwOnErr(cudaMemcpy(dIndices, indices.data(), indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dSink, totalThreads * sizeof(uint64_t)));
  throwOnErr(cudaMalloc(&dTimingResults, totalThreads * sizeof(uint64_t)));
  throwOnErr(cudaMalloc(&dSharedCycles, sizeof(uint64_t)));


  // Initialize events for timing
  cudaEvent_t evStart, evStop;
  throwOnErr(cudaEventCreate(&evStart));
  throwOnErr(cudaEventCreate(&evStop));
  throwOnErr(cudaEventRecord(evStart));


  auto kernel = getKernel<T>(mode);
  kernel<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
      dData, dIndices, data.size(), numAccesses, dTimingResults, dSharedCycles, dSink);


  // Measure using multiple SM blocks (host-side)
  throwOnErr(cudaEventRecord(evStop));
  throwOnErr(cudaEventSynchronize(evStop));
  float ms = 0.0f;
  throwOnErr(cudaEventElapsedTime(&ms, evStart, evStop));
  throwOnErr(cudaEventDestroy(evStart));
  throwOnErr(cudaEventDestroy(evStop));

  // throwOnErr(cudaDeviceSynchronize());
  throwOnErr(cudaGetLastError());

  // uint64_t* hTimingResults = static_cast<uint64_t*>(malloc(threadsPerBlock * numBlocks * sizeof(uint64_t)));
  // throwOnErr(cudaMemcpy(hTimingResults, dTimingResults, totalThreads * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // uint64_t hSharedCycles;
  // throwOnErr(cudaMemcpy(&hSharedCycles, dSharedCycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // cudaFree(dData);
  // cudaFree(dTimingResults);
  // cudaFree(dSink);

  // std::vector<uint64_t> result(totalThreads);
  // for (uint64_t i = 0; i < totalThreads; i++)
  //   result[i] = hTimingResults[i];
  // return {hSharedCycles, result};


  uint64_t* hTimingResults = static_cast<uint64_t*>(malloc(threadsPerBlock * numBlocks * sizeof(uint64_t)));
  throwOnErr(cudaMemcpy(hTimingResults, dTimingResults, totalThreads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  cudaFree(dData);
  cudaFree(dTimingResults);
  cudaFree(dSink);
  cudaFree(dSharedCycles);  // free this too

  // Convert CUDA event time (ms) to cycles using GPU clock frequency
  uint64_t hSharedCycles = static_cast<uint64_t>(ms * 1e-3 * getMaxClockFrequencyHz());
  
  // Build result vector from per-thread timing results
  std::vector<uint64_t> result(totalThreads);
  for (uint64_t i = 0; i < totalThreads; i++)
    result[i] = hTimingResults[i];
  
  free(hTimingResults);  // free the malloc'd buffer
  
  return {hSharedCycles, result};
}

/* The compiler complains when the concrete versions that we use aren't defined.
 * The implementation isn't required - just a header. So we declare a concrete
 * header for each version of randomAccessKernel that we use. */
template std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel<types::f8>(const std::vector<types::f8>&,
                                                                                        const std::vector<uint32_t>&,
                                                                                        uint64_t, uint64_t, uint64_t,
                                                                                        randomAccessKernel::mode);
template std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel<types::f16>(const std::vector<types::f16>&,
                                                                                         const std::vector<uint32_t>&,
                                                                                         uint64_t, uint64_t, uint64_t,
                                                                                         randomAccessKernel::mode);
template std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel<types::f32>(const std::vector<types::f32>&,
                                                                                         const std::vector<uint32_t>&,
                                                                                         uint64_t, uint64_t, uint64_t,
                                                                                         randomAccessKernel::mode);
template std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel<types::f64>(const std::vector<types::f64>&,
                                                                                         const std::vector<uint32_t>&,
                                                                                         uint64_t, uint64_t, uint64_t,
                                                                                         randomAccessKernel::mode);
