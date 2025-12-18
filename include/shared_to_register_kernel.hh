#ifndef SHARED_TO_REGISTER_KERNEL_HH
#define SHARED_TO_REGISTER_KERNEL_HH

#include <cstdint>
#include <stdexcept>
#include <string>

#include <Common.hh>

namespace sharedToRegister {

enum mode {
  READ,
  WRITE,
  READ_WRITE,
};

static const constexpr std::string_view modeRead = "read";
static const constexpr std::string_view modeWrite = "write";
static const constexpr std::string_view modeReadWrite = "read_write";

static __used mode parseMode(std::string& modeArg) {
  if (modeArg == modeRead)
    return READ;
  else if (modeArg == modeWrite)
    return WRITE;
  else if (modeArg == modeReadWrite)
    return READ_WRITE;
  throw std::invalid_argument("parseMode: invalid mode " + modeArg);
}

} /* namespace sharedToRegister */

/**
 * Launch the shared memory kernel.
 *
 * @param numElems Number of 32-bit words in shared memory, must be power of two.
 * @param numIters Iterations per thread of the measured loop.
 * @param threads Threads per block.
 * @param numBlocks Number of blocks to launch.
 * @param sharedBytes Size in bytes of the extern shared memory.
 * @param stride Stride in words used to create bank conflict patterns.
 * @param mode The access mode (READ, WRITE, READ_WRITE).
 * @param elapsedMs Returned elapsed milliseconds for measured kernel.
 */
void launchSharedToRegisterKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint32_t numBlocks,
                                  uint64_t sharedBytes, uint32_t stride, sharedToRegister::mode mode,
                                  float* elapsedMs);

#endif /* SHARED_TO_REGISTER_KERNEL_HH */