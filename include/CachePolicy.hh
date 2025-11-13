/**
 * @brief Defines cache hierarchy levels for bandwidth measurement.
 *
 * Used to specify which memory region (L1, L2, or DRAM) is targeted
 * during cache or memory access benchmarks.
 */
#pragma once

namespace cacheload {

enum class CachePolicy { L1, L2, DRAM };

}
