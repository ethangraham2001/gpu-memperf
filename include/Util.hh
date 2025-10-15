#ifndef UTIL_HH
#define UTIL_HH

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

}  // namespace util

#endif /* UTIL_HH */
