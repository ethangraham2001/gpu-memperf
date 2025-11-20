/**
 * Strided L1 bandwidth kernel launcher.
 *
 * Provides a host-callable wrapper around a CUDA kernel that performs many
 * strided global memory loads intended to hit in L1 and measure per-SM/device
 * bandwidth. Timing is done with CUDA events and returned in milliseconds.
 */
#ifndef STRIDED_ACCESS_HH
#define STRIDED_ACCESS_HH

#include <cstdint>
#include <vector>

#include <CachePolicy.hh>

/** Host-callable wrapper. Times the kernel with CUDA events. */
template <typename T, cacheload::CachePolicy>
float launchStridedAccessKernel(const std::vector<T>& data, uint64_t stride, uint64_t iters, uint64_t threadsPerBlock,
                                uint64_t numBlocks);

#endif /* STRIDED_ACCESS_HH */
