#include <cuda.h>
#include <cassert>
#include <vector>

#include <Types.hh>
#include <cudaHelpers.cuh>
#include <random_access_kernel.hh>

#include <clock64.hh>

/**
 * l1LoadElem - inline PTX for loading an element from L1 cache
 *
 * Constexpr templating means that this if statement is never evaluated at
 * runtime on the device - it is evaluated statically.
 * 
 * Using ca (cache all) modifier to prioritize L1 cache.
 */
template <typename T>
__device__ __forceinline__ void l1LoadElem(T* addr, T& sink) {
  if constexpr (sizeof(T) == sizeof(types::f32)) {
    asm volatile("{\t\n .reg .f32 data_reg;\n\t ld.global.ca.f32 data_reg, [%1];\n\t add.f32 %0, %0, data_reg;\n\t }"
                 : "+f"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f64)) {
    asm volatile("{\t\n .reg .f64 data_reg;\n\t ld.global.ca.f64 data_reg, [%1];\n\t add.f64 %0, %0, data_reg;\n\t }"
                 : "+d"(sink)
                 : "l"(addr)
                 : "memory");
  }
}

/**
 * l2LoadElem - inline PTX for loading an element from L2 cache
 *
 * Using cg (cache global) modifier to bypass L1 cache.
 */
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
 * dramLoadElem - inline PTX for loading an element from DRAM
 *
 * Using cv (cache volatile) modifier to bypass all caches.
 */
template <typename T>
__device__ __forceinline__ void dramLoadElem(T* addr, uint64_t& sink) {
  if constexpr (sizeof(T) == sizeof(types::f8)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cv.u8 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f16)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cv.u16 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f32)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cv.u32 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  } else if constexpr (sizeof(T) == sizeof(types::f64)) {
    asm volatile("{\t\n .reg .u64 data64;\n\t ld.global.cv.u64 data64, [%1];\n\t add.u64 %0, %0, data64;\n\t }"
                 : "+l"(sink)
                 : "l"(addr)
                 : "memory");
  }
}

/** 
 * Loader structs to wrap load functions at compile time - L1, L2, DRAM
 */
template <typename T>
struct L1Loader {
  __device__ __forceinline__ void operator()(T* addr, uint64_t& sink) const { l1LoadElem<T>(addr, sink); }
};

template <typename T>
struct L2Loader {
  __device__ __forceinline__ void operator()(T* addr, uint64_t& sink) const { l2LoadElem<T>(addr, sink); }
};

template <typename T>
struct DramLoader {
  __device__ __forceinline__ void operator()(T* addr, uint64_t& sink) const { dramLoadElem<T>(addr, sink); }
};

/**
 * accumulateRandomAccesses - shared access pattern for warmup and benchmark kernels
 * 
 * @tid: thread ID
 * @totalThreads: total number of threads
 * @data: data array
 * @indices: random index array
 * @numElems: number of elements in data/indices
 * @numAccesses: number of accesses per thread
 * @loadFunc: function to load an element
 */
template <typename T, typename LoadFunc>
__device__ __forceinline__ uint64_t accumulateRandomAccesses(uint64_t tid, uint64_t totalThreads, T* data,
                                                             uint32_t* indices, uint64_t numElems, uint64_t numAccesses,
                                                             LoadFunc loadFunc) {
  uint64_t localSink = 0;
  for (uint64_t i = 0; i < numAccesses; i++) {
    uint64_t idx = indices[(tid + i * totalThreads) mod_power_of_2(numElems)];
    loadFunc(&data[idx], localSink);
  }
  return localSink;
}

/**
 * randomAccessKernelDispatch - dispatch kernel to select loader at compile time
 * 
 * Legacy specialized version kept for reference while getKernel handles loader dispatch.
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
 * @totalCycles: cycles for all threads to complete as measured by thread 0
 * @sink[numThreads]: used to prevent compiler optimization - can be ignored
 */
template <typename T, template <typename> class Loader>
__global__ void randomAccessKernelDispatch(T* data, uint32_t* indices, uint64_t numElems, uint64_t numAccesses,
                                           uint64_t* sink) {
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

/**
 * Type aliases for function pointers to kernels
 */
template <typename T>
using randomAccessKernelFunc = void (*)(T*, uint32_t*, uint64_t, uint64_t, uint64_t*);

/**
 * Type aliases for function pointers to warmup kernels
 */
template <typename T>
using randomAccessWarmupKernelFunc = void (*)(T*, uint32_t*, uint64_t, uint64_t, uint64_t*);

/**
 * getKernel - select the appropriate kernel based on mode
 * 
 * @mode: cache mode
 * @return: function pointer to the selected kernel
 */
template <typename T>
static randomAccessKernelFunc<T> getKernel(randomAccessKernel::mode mode) {
  switch (mode) {
    case randomAccessKernel::L1_CACHE:
      return randomAccessKernelDispatch<T, L1Loader>;
    case randomAccessKernel::L2_CACHE:
      return randomAccessKernelDispatch<T, L2Loader>;
    case randomAccessKernel::DRAM:
      return randomAccessKernelDispatch<T, DramLoader>;
    default:
      throw std::invalid_argument("invalid mode");
  }
}

/**
 * getWarmupKernel - select the appropriate warmup kernel based on mode
 * 
 * @mode: cache mode
 * @return: function pointer to the selected warmup kernel
 */
template <typename T>
static randomAccessWarmupKernelFunc<T> getWarmupKernel(randomAccessKernel::mode mode) {
  switch (mode) {
    case randomAccessKernel::L1_CACHE:
      return randomAccessWarmupDispatch<T, L1Loader>;
    case randomAccessKernel::L2_CACHE:
      return randomAccessWarmupDispatch<T, L2Loader>;
    case randomAccessKernel::DRAM:
      return randomAccessWarmupDispatch<T, DramLoader>;
    default:
      throw std::invalid_argument("getWarmupKernel: invalid mode");
  }
}

/**
 * launchRandomAccessKernel - launch the random access kernel and measure time
 *
 * @data: data array
 * @indices: random index array
 * @numAccesses: number of accesses per thread
 * @threadsPerBlock: number of threads per block
 * @numBlocks: number of blocks
 * @mode: cache mode
 * @return: total cycles taken for the kernel to complete
 */
template <typename T>
uint64_t launchRandomAccessKernel(const std::vector<T>& data, const std::vector<uint32_t>& indices,
                                  uint64_t numAccesses, uint64_t threadsPerBlock, uint64_t numBlocks,
                                  randomAccessKernel::mode mode) {

  uint64_t numElems = data.size();
  uint64_t numIndices = indices.size();
  assert(isPowerOf2(numElems));
  assert(numElems == numIndices);

  uint32_t* dIndices;
  uint64_t* dSink;
  T* dData;

  uint64_t totalThreads = numBlocks * threadsPerBlock;

  throwOnErr(cudaMalloc(&dData, numElems * sizeof(T)));
  throwOnErr(cudaMemcpy(dData, data.data(), numElems * sizeof(T), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dIndices, numIndices * sizeof(uint32_t)));
  throwOnErr(cudaMemcpy(dIndices, indices.data(), numIndices * sizeof(uint32_t), cudaMemcpyHostToDevice));

  throwOnErr(cudaMalloc(&dSink, totalThreads * sizeof(uint64_t)));

  auto warmupKernel = getWarmupKernel<T>(mode);
  if (warmupKernel) {
    warmupKernel<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
        dData, dIndices, data.size(), numAccesses, dSink);
    throwOnErr(cudaDeviceSynchronize());
  }

  /* Initialize events for timing. */
  cudaEvent_t evStart, evStop;
  throwOnErr(cudaEventCreate(&evStart));
  throwOnErr(cudaEventCreate(&evStop));
  throwOnErr(cudaEventRecord(evStart));

  auto kernel = getKernel<T>(mode);
  kernel<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
      dData, dIndices, numElems, numAccesses, dSink);

  /* Measure using multiple thread blocks (host-side). */
  throwOnErr(cudaEventRecord(evStop));
  throwOnErr(cudaEventSynchronize(evStop));
  float ms = 0.0f;
  throwOnErr(cudaEventElapsedTime(&ms, evStart, evStop));
  throwOnErr(cudaEventDestroy(evStart));
  throwOnErr(cudaEventDestroy(evStop));

  throwOnErr(cudaGetLastError());

  cudaFree(dData);
  cudaFree(dIndices);
  cudaFree(dSink);

  /* Convert CUDA event time (ms) to cycles using GPU clock frequency */
  uint64_t hSharedCycles = static_cast<uint64_t>(ms * 1e-3 * getMaxClockFrequencyHz());
  // TODO: decide if we keep results in cycles or ms (above - currently in cycles)
  return hSharedCycles;
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
