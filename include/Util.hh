#ifndef UTIL_HH
#define UTIL_HH

#include <cstdint>
#include <numeric>
#include <ostream>
#include <vector>

namespace util {

template <typename T>
void displayVector(std::ostream& os, const std::vector<T>& vec) {
  os << "[\n";
  for (const auto& elem : vec) {
    os << "\t" << elem << ",\n";
  }
  os << "]" << std::endl;
}

/**
 * makeCoprime - increment a until it is coprime with b
 */
static uint64_t makeCoprime(uint64_t a, uint64_t b) {
  while (std::gcd(a, b) != 1)
    a++;
  return a;
}

}  // namespace util

#endif /* UTIL_HH */
