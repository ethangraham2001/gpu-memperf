#include <cassert>

#include <cuda_runtime.h>

#include <cudaHelpers.cuh>
#include <device_attributes.hh>
#include <shared_to_register_kernel.hh>

template <sharedToRegister::mode MODE>
__global__ void sharedToRegisterKernel(uint32_t numElems, uint32_t numIters, uint32_t stride) {
  const uint32_t tid = threadIdx.x;
  const uint32_t mask = numElems - 1u;

  extern __shared__ uint64_t sharedMem[];
  uint64_t tmp = 0u;

  for (uint32_t i = tid; i < numElems; i += blockDim.x) {
    sharedMem[i] = i;
  }
  __syncthreads();

  /* Loop over shared memory with given stride. */
  for (uint32_t i = 0; i < numIters; ++i) {
    uint32_t offset = (tid % 32) * stride;  /* Bank offset */
    uint32_t address = (i + offset) & mask; /* Prevent optimization between iterations */

    if constexpr (MODE == sharedToRegister::READ) {
      tmp += sharedMem[address];
    } else if constexpr (MODE == sharedToRegister::WRITE) {
      sharedMem[address] = tid + i;
    } else {
      uint64_t v = sharedMem[address];
      sharedMem[address] = v + 1;
    }
  }

  /* Prevent compiler optimization. */
  if (tmp == 0xFFFFFFFFu) {
    sharedMem[0] = tmp;
  }
}

template <sharedToRegister::mode MODE>
void launchKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint32_t numBlocks, uint64_t sharedBytes,
                  uint32_t stride, float* elapsedMs) {
  cudaError_t err;
  const dim3 block(threads);
  const dim3 grid(numBlocks);
  const uint32_t warmupIters = 1000;

  /* Warmup launch. */
  sharedToRegisterKernel<MODE><<<grid, block, sharedBytes>>>(numElems, warmupIters, stride);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Kernel uses extern shared memory size = sharedBytes. */
  cudaEventRecord(start);
  sharedToRegisterKernel<MODE><<<grid, block, sharedBytes>>>(numElems, numIters, stride);
  cudaEventRecord(stop);

  err = cudaEventSynchronize(stop);
  throwOnErr(err);

  float ms = 0.0f;
  err = cudaEventElapsedTime(&ms, start, stop);
  throwOnErr(err);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  err = cudaGetLastError();
  throwOnErr(err);

  if (elapsedMs)
    *elapsedMs = ms;
}

void launchSharedToRegisterKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint32_t numBlocks,
                                  uint64_t sharedBytes, uint32_t stride, sharedToRegister::mode mode,
                                  float* elapsedMs) {
  assert(isPowerOf2(numElems));

  switch (mode) {
    case sharedToRegister::READ:
      launchKernel<sharedToRegister::READ>(numElems, numIters, threads, numBlocks, sharedBytes, stride, elapsedMs);
      break;
    case sharedToRegister::WRITE:
      launchKernel<sharedToRegister::WRITE>(numElems, numIters, threads, numBlocks, sharedBytes, stride, elapsedMs);
      break;
    case sharedToRegister::READ_WRITE:
      launchKernel<sharedToRegister::READ_WRITE>(numElems, numIters, threads, numBlocks, sharedBytes, stride, elapsedMs);
      break;
    default:
      throw std::runtime_error("Invalid mode");
  }
}
