#ifndef GLOBAL_TO_SHARED_HH
#define GLOBAL_TO_SHARED_HH

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Common.hh>
#include "Types.hh"

namespace globalToShared {
enum mode {
  SYNC,
  ASYNC_2X_BUFFERED,
  ASYNC_4X_BUFFERED,
};

static __used std::string modeStr(mode modeArg) {
  switch (modeArg) {
    case SYNC:
      return "sync";
    case ASYNC_2X_BUFFERED:
      return "async_2x";
    case ASYNC_4X_BUFFERED:
      return "async_4x";
  }
  throw std::invalid_argument("invalid mode");
}

} /* namespace globalToShared */

template <uint32_t TileSize>
float launchGlobalToSharedKernel(globalToShared::mode mode, const std::vector<types::f32>& globalBuffer,
                                 uint64_t numFlops, uint64_t threadsPerBlock, uint64_t numBlocks);

#endif /* GLOBAL_TO_SHARED_HH */
