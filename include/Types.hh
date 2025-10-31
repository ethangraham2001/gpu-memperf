#ifndef TYPES_HH
#define TYPES_HH

#include <cstdint>

/*
 * None of our benchmarks should perform any computation. So if we want an f8
 * on the device, we can just generate an integer and treat it as a float on
 * the GPU without loss of generality. It's just a one-byte wide piece of data.
 */
namespace types {
using f8 = uint8_t;
using f16 = uint16_t;
using f32 = uint32_t;
using f64 = uint64_t;
};  // namespace types

#endif /* TYPES_HH */
