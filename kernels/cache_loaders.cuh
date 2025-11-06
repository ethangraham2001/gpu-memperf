#pragma once
#include <cuda_runtime.h>
#include "CachePolicy.hh"
#include "Types.hh"

namespace cacheload {

/* Load using specific cache policy P and element type T.
   Uses sizeof(T) == sizeof(types::f*) to pick PTX width. */
template <CachePolicy P, typename T>
__device__ __forceinline__ void loadElem(const T* addr, uint64_t& sink) {
  if constexpr (P == CachePolicy::L1) { /* L1 */
    if constexpr (sizeof(T) == sizeof(types::f8)) {
      asm volatile("{ .reg .u64 r; ld.global.ca.u8  r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f16)) {
      asm volatile("{ .reg .u64 r; ld.global.ca.u16 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f32)) {
      asm volatile("{ .reg .u64 r; ld.global.ca.u32 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f64)) {
      asm volatile("{ .reg .u64 r; ld.global.ca.u64 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    }
  } else if constexpr (P == CachePolicy::L2) { /* L2 */
    if constexpr (sizeof(T) == sizeof(types::f8)) {
      asm volatile("{ .reg .u64 r; ld.global.cg.u8  r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f16)) {
      asm volatile("{ .reg .u64 r; ld.global.cg.u16 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f32)) {
      asm volatile("{ .reg .u64 r; ld.global.cg.u32 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f64)) {
      asm volatile("{ .reg .u64 r; ld.global.cg.u64 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    }
  } else { /* DRAM */
    if constexpr (sizeof(T) == sizeof(types::f8)) {
      asm volatile("{ .reg .u64 r; ld.global.cv.u8  r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f16)) {
      asm volatile("{ .reg .u64 r; ld.global.cv.u16 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f32)) {
      asm volatile("{ .reg .u64 r; ld.global.cv.u32 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    } else if constexpr (sizeof(T) == sizeof(types::f64)) {
      asm volatile("{ .reg .u64 r; ld.global.cv.u64 r, [%1]; add.u64 %0,%0,r; }" : "+l"(sink) : "l"(addr) : "memory");
    }
  }
}

template <CachePolicy P>
struct Loader {
  template <typename T>
  __device__ __forceinline__ void operator()(const T* addr, uint64_t& sink) const {
    loadElem<P>(addr, sink);
  }
};

/* Aliases for convenience */
using L1Loader = Loader<CachePolicy::L1>;
using L2Loader = Loader<CachePolicy::L2>;
using DramLoader = Loader<CachePolicy::DRAM>;

}  // namespace cacheload
