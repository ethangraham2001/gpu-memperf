#ifndef PCHASEGPUBENCHMARK_HH
#define PCHASEGPUBENCHMARK_HH

#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>

/* Forward declaration. */
void launchPChaseKernel(uint64_t* array, uint64_t arraySize, uint64_t iters, uint64_t* total_cycles);

/**
 * PchaseGPUBenchmark - use the PChase algorithm to estimate GPU cache sizes
 *
 * As presented in the Citadel paper[1], the nominal cache sizes are likely
 * greater than the detectable cache sizes. For this reason, it is useful to
 * have a benchmark that measures average cache latency as a means to estimate
 * the effective cache sizes of the GPUs that we work with and as a result
 * write bandwidth benchmarks that are tailored for the actual cache size.
 *
 * [1] https://arxiv.org/pdf/1804.06826
 */
class PChaseGPUBenchmark {
 public:
  static constexpr const char* benchmarkName = "pchase_gpu";

  PChaseGPUBenchmark(const std::vector<std::string>& args = {}) {
    benchmark::ArgParser parser(args);
    numExperiments_ = parser.getOr("num_experiments", 12);
    multiplier_ = parser.getOr("multiplier", 2);
    numIters_ = parser.getOr("num_iters", 1000000UL);
    startBytes_ = parser.getOr("start_bytes", 1 << 16 /* 65536. */);
    stride_ = parser.getOr("stride", 1 << 6);
  }

  std::string name() const { return benchmarkName; }

  void run(std::ostream& os) {
    const auto result = runBenchmarks(startBytes_, multiplier_, numExperiments_, numIters_);
    os << "bytes,avg_access_latency\n";
    for (const auto& [bytes, avgLatency] : result)
      os << bytes << "," << avgLatency << "\n";
  }

 private:
  uint64_t numExperiments_;
  uint64_t multiplier_;
  uint64_t numIters_;
  uint64_t startBytes_;
  uint64_t stride_; /* Stride in array elements, where each element is 8 bytes. */

  static uint64_t* initializeChain(uint64_t numEntries, uint64_t stride) {
    uint64_t* chain = static_cast<uint64_t*>(malloc(numEntries * sizeof(uint64_t)));
    if (!chain)
      throw std::bad_alloc();
    for (uint64_t i = 0; i < numEntries; i++) {
      chain[i] = (i + stride) % numEntries;
    }
    return chain;
  }

  static void releaseChain(uint64_t* chain) { free(chain); }

  /** Run a benchmark, return the total number of accesses. */
  static uint64_t runExperiment(uint64_t numIters, uint64_t stride, uint64_t numBytes) {
    if (numBytes % sizeof(uint64_t))
      throw std::runtime_error("number of bytes should be a multiple of 8 bytes");

    uint64_t numEntries = numBytes / sizeof(uint64_t);
    uint64_t* chain = initializeChain(numEntries, stride);

    uint64_t cycles;
    launchPChaseKernel(chain, numBytes, numIters, &cycles);

    releaseChain(chain);
    return cycles;
  }

  const std::vector<std::pair<uint64_t, double>> runBenchmarks(uint64_t startBytes, uint64_t multiplier,
                                                               uint64_t numExperiments, uint64_t numIters) const {
    std::vector<std::pair<uint64_t, double>> out;
    out.reserve(numExperiments_);
    uint64_t numBytes = startBytes;
    for (uint64_t i = 0; i < numExperiments; i++) {
      uint64_t cycles = runExperiment(numIters, stride_, numBytes);
      double avgCyclesPerAccess = (double)cycles / (double)numIters;
      out.push_back({numBytes, avgCyclesPerAccess});
      numBytes *= multiplier;
    }
    return out;
  }
};

inline bool pchaseGPUBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<PChaseGPUBenchmark>(PChaseGPUBenchmark::benchmarkName);
  return true;
}();

#endif /* PCHASEGPUBENCHMARK_HH */
