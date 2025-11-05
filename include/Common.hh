#ifndef COMMON_HH
#define COMMON_HH

#include <cstdint>

#define __used __attribute__((used))

namespace common {
static const constexpr uint64_t KiB = 1UL << 10;
static const constexpr uint64_t MiB = 1UL << 20;
static const constexpr uint64_t GiB = 1UL << 30;

static const constexpr uint64_t threadsPerWarp = 32UL;
};  // namespace common

#endif /* COMMON_HH */
