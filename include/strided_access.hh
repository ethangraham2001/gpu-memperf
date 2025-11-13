/**
 * Strided L1 bandwidth kernel launcher.
 *
 * Provides a host-callable wrapper around a CUDA kernel that performs many
 * strided global memory loads intended to hit in L1 and measure per-SM/device
 * bandwidth. Timing is done with CUDA events and returned in milliseconds.
 */
#ifndef STRIDED_ACCESS_HH
#define STRIDED_ACCESS_HH
#include "CachePolicy.hh"

#include <cstdint>

/** Host-callable wrapper. Times the kernel with CUDA events. */
template <typename T>
void launchStridedAccessKernel(const std::vector<T>& data, cacheload::CachePolicy policy, uint64_t stride,
                               uint64_t iters, uint64_t threadsPerBlock, uint64_t numBlocks, uint64_t* totalCycles);

#endif /* STRIDED_ACCESS_HH */
