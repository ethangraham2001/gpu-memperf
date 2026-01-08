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
    8 * common::GiB,   /* DRAM. */
};

/**
 * sensibleNumBlocks - return a sensible default number of blocks for each mode
 *
 * The default number of blocks is set to 36 for all memory levels, for a one-to-one mapping with SMs.
 */
static const constexpr uint64_t sensibleNumBlocks[MODES_SIZE] = {
    36, /* L1 Cache. */
    36, /* L2 Cache. */
    36, /* DRAM. */
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
uint64_t launchRandomAccessKernel(const std::vector<T>& data, const std::vector<uint32_t>& indices,
                                  uint64_t numAccesses, uint64_t threadsPerBlock, uint64_t numBlocks,
                                  randomAccessKernel::mode mode);

#endif /* RANDOM_ACCESS_KERNEL_HH */
