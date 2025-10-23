#ifndef CLOCK64_HH
#define CLOCK64_HH

#include <cstdint>

/** Host-callable wrapper. */
double measureClock64Latency(uint64_t iters);

#endif /* CLOCK64_HH */
