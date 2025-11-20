#ifndef CLOCK64_HH
#define CLOCK64_HH

#include <cstdint>

/** Host-callable wrapper. */
double measureClock64Latency(uint64_t iters);

/** Returns the maximum clock frequency in Hz. */
unsigned int getMaxClockFrequencyHz();

/** Returns the number of SMs on device 0 */
unsigned int getSmCount(int device);

#endif /* CLOCK64_HH */
