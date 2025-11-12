#ifndef RANDOM_ACCESS_KERNEL_HH
#define RANDOM_ACCESS_KERNEL_HH

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Common.hh>

namespace randomAccessKernel {
enum mode {
  L1_CACHE,
  L2_CACHE,
  DRAM,
  MODES_SIZE,
};

static const constexpr std::string_view modeL1 = "l1";
static const constexpr std::string_view modeL2 = "l2";
static const constexpr std::string_view modeDram = "dram";

static const constexpr uint64_t sensibleCacheSizes[MODES_SIZE] = {
    100 * common::KiB, /* L1. */
    25 * common::MiB,  /* L2. */
    2 * common::GiB,   /* DRAM. */
};

static const constexpr uint64_t sensibleNumBlocks[MODES_SIZE] = {
    1,  /* L1. */
    64, /* L2. Gave the highest and most stable L2 bandwidth without unnecessary scheduling overhead */
    10, /* DRAM. TODO: figure out what this should be. */
};

static __used mode parseMode(std::string& modeArg) {
  if (modeArg == randomAccessKernel::modeL1)
    return randomAccessKernel::L1_CACHE;
  else if (modeArg == randomAccessKernel::modeL2)
    return randomAccessKernel::L2_CACHE;
  else if (modeArg == randomAccessKernel::modeDram)
    return randomAccessKernel::DRAM;
  throw std::invalid_argument("parseMode: invalid mode " + modeArg);
}

/**
 * defaultCacheSize - return a conservative size estimate of a cache level
 */
static __used uint64_t defaultCacheSize(mode m) {
  if (m < 0 || m >= MODES_SIZE)
    throw std::invalid_argument("invalid mode");
  return sensibleCacheSizes[static_cast<int>(m)];
}

static __used uint64_t defaultNumBlocks(mode m) {
  if (m < 0 || m >= MODES_SIZE)
    throw std::invalid_argument("invalid mode");
  return sensibleNumBlocks[static_cast<int>(m)];
}

} /* namespace randomAccessKernel */

/** Host-callable wrapper. */
template <typename T>
std::pair<uint64_t, std::vector<uint64_t>> launchRandomAccessKernel(const std::vector<T>& data,
                                                                    const std::vector<uint32_t>& indices,
                                                                    uint64_t numAccesses, uint64_t threadsPerBlock,
                                                                    uint64_t numBlocks, randomAccessKernel::mode mode);

#endif /* RANDOM_ACCESS_KERNEL_HH */
