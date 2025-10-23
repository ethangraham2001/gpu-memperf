#ifndef PCHASE_HH
#define PCHASE_HH

#include <cstdint>

/** Host-callable wrapper. */
void launchPChaseKernel(uint64_t* array, uint64_t arraySize, uint64_t iters, uint64_t* total_cycles);

#endif /* PCHASE_HH */
