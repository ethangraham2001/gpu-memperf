#ifndef BENCHMARK_HH
#define BENCHMARK_HH

#include <algorithm>
#include <concepts>
#include <functional>
#include <map>
#include <ostream>
#include <vector>

#include <Encoder.hh>

namespace benchmark {

/**
 * Benchmark - represents the high-level behavior of a benchmark
 *
 * A benchmark is some structure with the following characteristics
 *
 * - Its constructor should take a list of arguments.
 * - It must be named.
 * - It should be runnable, writing its output to some output stream.
 */
template <typename T>
concept Benchmark = requires(T bench, Encoder& e, const std::vector<std::string>& args) {
  // clang-format off
  {bench.name()}->std::convertible_to<std::string>;
  {bench.run()}->std::same_as<void>;
  // clang-format on
};

/**
 * BenchmarkRegistry - centralizes the set of benchmarks defined by gpu-memperf
 *
 * Benchmarks should be registered in a static inline function so that they are
 * initialized before launch, and accessible by the registry at runtime.
 */
class BenchmarkRegistry {
  std::map<std::string, std::function<void(Encoder& enc, const std::vector<std::string>&)>> benchmarks_;

 public:
  static BenchmarkRegistry& instance() {
    static BenchmarkRegistry reg;
    return reg;
  }

  template <Benchmark B>
  void registerBenchmark(std::string name) {
    benchmarks_[name] = [](Encoder& enc, const std::vector<std::string>& args) {
      B bench(enc, args);
      bench.run();
    };
  }

  void run(const std::string& name, Encoder& enc, const std::vector<std::string>& args) {
    benchmarks_.at(name)(enc, args);
  }

  bool exists(const std::string& name) const { return benchmarks_.contains(name); }

  const std::vector<std::string> availableBenchmarks() const {
    std::vector<std::string> benches;
    std::transform(benchmarks_.begin(), benchmarks_.end(), std::back_inserter(benches),
                   [](const auto& pair) { return pair.first; });
    return benches;
  }
};

};  // namespace benchmark

#endif /* BENCHMARK_HH */
