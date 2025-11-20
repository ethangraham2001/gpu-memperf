#include <cassert>

#include <cuda_runtime.h>

#include <cudaHelpers.cuh>
#include <device_attributes.hh>
#include <shared_to_register_kernel.hh>

template <sharedToRegister::mode MODE>
__global__ void sharedToRegisterKernel(uint32_t numElems, uint32_t numIters, uint32_t stride, uint64_t* cycle) {
  const uint32_t tid = threadIdx.x;
  const uint32_t mask = numElems - 1u;

  extern __shared__ uint32_t sharedMem[];
  uint32_t tmp = 0u;

  for (uint32_t i = tid; i < numElems; i += blockDim.x) {
    sharedMem[i] = i;
  }
  __syncthreads();

  uint64_t start = 0, end = 0;

  if (tid == 0)
    start = clock64();

  /* Loop over shared memory with given stride. */
  for (uint32_t i = 0; i < numIters; ++i) {
    uint32_t offset = ((tid + i) * stride) & mask;

    if constexpr (MODE == sharedToRegister::READ) {
      tmp += sharedMem[offset];
    } else if constexpr (MODE == sharedToRegister::WRITE) {
      sharedMem[offset] = tid + i;
    } else {
      uint64_t v = sharedMem[offset];
      sharedMem[offset] = v + 1;
    }
  }

  /* Prevent compiler optimization. */
  if (tmp == 0xFFFFFFFFu) {
    sharedMem[0] = tmp;
  }
  __syncthreads();

  if (tid == 0) {
    end = clock64();
    *cycle = end - start;
  }
}

template <sharedToRegister::mode MODE>
void launchKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes, uint32_t stride,
                  uint64_t* cycle) {
  cudaError_t err;
  const dim3 block(threads);
  const uint32_t warmupIters = 256;
  uint64_t warmupCycle = 0;

  /* Warmup launch. */
  sharedToRegisterKernel<MODE><<<1, block, sharedBytes>>>(numElems, warmupIters, stride, &warmupCycle);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  /* Kernel uses extern shared memory size = sharedBytes. */
  sharedToRegisterKernel<MODE><<<1, block, sharedBytes>>>(numElems, numIters, stride, cycle);
  err = cudaDeviceSynchronize();
  throwOnErr(err);

  err = cudaGetLastError();
  throwOnErr(err);
}

void launchSharedToRegisterKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                  uint32_t stride, sharedToRegister::mode mode, uint64_t* cycle) {
  assert(isPowerOf2(numElems));

  switch (mode) {
    case sharedToRegister::READ:
      launchKernel<sharedToRegister::READ>(numElems, numIters, threads, sharedBytes, stride, cycle);
      break;
    case sharedToRegister::WRITE:
      launchKernel<sharedToRegister::WRITE>(numElems, numIters, threads, sharedBytes, stride, cycle);
      break;
    case sharedToRegister::READ_WRITE:
      launchKernel<sharedToRegister::READ_WRITE>(numElems, numIters, threads, sharedBytes, stride, cycle);
      break;
    default:
      throw std::runtime_error("Invalid mode");
  }
}
