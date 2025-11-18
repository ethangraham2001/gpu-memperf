#ifndef UTIL_HH
#define UTIL_HH

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
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

/**
 * permutation - return a permutation of range [0, n - 1]
 */
template <typename T>
static const std::vector<T> permutation(uint64_t n) {
  std::vector<T> out(n);
  std::iota(out.begin(), out.end(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(out.begin(), out.end(), gen);
  return out;
}

template <typename T>
uint64_t bytesToNumElems(uint64_t bytes) {
  if (bytes % sizeof(T))
    throw std::invalid_argument("bytes can not be perfectly divided");
  return bytes / sizeof(T);
}

/**
 * formatBytes - format some number of bytes in human readable form
 *
 * For example, 1.52 billion bytes would be formatted as 1.5GB
 */
static std::string formatBytes(double numBytes) {
  std::string fmt;
  if (numBytes < 1e3) {
    fmt = "B";
  } else if (numBytes < 1e6) {
    fmt = "KB";
    numBytes /= 1e3;
  } else if (numBytes < 1e9) {
    fmt = "MB";
    numBytes /= 1e6;
  } else if (numBytes < 1e12) {
    fmt = "GB";
    numBytes /= 1e9;
  } else if (numBytes < 1e15) {
    fmt = "TB";
    numBytes /= 1e12;
  } else {
    throw std::runtime_error(std::string(__FUNCTION__) + ": too many bytes to format");
  }
  return std::to_string(numBytes) + fmt;
}

}  // namespace util

#endif /* UTIL_HH */
