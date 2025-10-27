#ifndef PCHASEGPUBENCHMARK_HH
#define PCHASEGPUBENCHMARK_HH

#include <map>
#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include "Util.hh"

#include <clock64.hh>
#include <pchase.hh>

/**
 * PchaseGPUBenchmark - use the PChase algorithm to estimate GPU cache sizes
 *
 * As presented in the Citadel paper[1], the nominal cache sizes are likely
 * greater than the detectable cache sizes. For this reason, it is useful to
 * have a benchmark that measures average cache latency as a means to estimate
 * the effective cache sizes of the GPUs that we work with and as a result
 * write bandwidth benchmarks that are tailored for the actual cache size.
 *
 * The algorithm consists of several steps.
 * 1. A coarse sweep, starting at `startBytes_`, and multiplying the size of
 *    the pointer chain by `multiplier_` every iteration. This unveils coarse
 *    patterns in access latency, approximating where cache size boundaries
 *    exist.
 * 2. Ridge detection using the coarse grain metrics. A ridge is identified by
 *    a `coarseThresh_` relative increase in latency between two measurements.
 * 3. A fine-grained sweep over the regions around the ridges, providing higher
 *    resolution metrics over the "interesting" regions.
 *
 * NOTE: it's important that the stride parameter be coprime with the number
 * of elements in the chain array. For example, assume a stride of 64 and a
 * chain array of size 128. We would end up exclusively accessing elements
 * 0 and 64 before wrapping back around, rendering most of the array totally
 * useless.
 *
 * [1] https://arxiv.org/pdf/1804.06826
 */
class PChaseGPUBenchmark {
 public:
  static constexpr const char* benchmarkName = "pchase_gpu";

  PChaseGPUBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    numExperiments_ = parser.getOr("num_experiments", 12UL);
    multiplier_ = parser.getOr("multiplier", 2UL);
    numIters_ = parser.getOr("num_iters", 1000000UL);
    startBytes_ = parser.getOr("start_bytes", 1UL << 16 /* 65536. */);
    /* We choose 16 as a default stride, as 16 * 8B = 128, which is the cache
     * line size on most modern NVIDIA GPUs. */
    stride_ = parser.getOr("stride", 16UL);
    /* 20% increases in latency are interpreted as a potential cache size
     * boundary during the coarse-grained sweep. */
    threshCoarse_ = parser.getOr("thresh_coarse", 1.2);
  }

  std::string name() const { return benchmarkName; }

  void run() {
    /* Step 1: take coarse grained measurements, multiplying the effective size
     * of the chain by 2 every iteration. This detects approximately where large
     * increases in access latency occur. */
    const auto coarse = runCoarse(startBytes_, multiplier_, numExperiments_);

    /* Step 2: take finer grained measurements around the areas where we saw
     * a latency increase. */
    const auto ridges = findRidges(coarse, threshCoarse_);
    std::vector<std::pair<uint64_t, double>> fine;
    for (const auto& [lb, ub] : ridges) {
      uint64_t stepSize = (ub - lb) / 32;
      const auto fineGrained = runFine(lb, ub, stepSize);
      for (const auto& p : fineGrained)
        fine.push_back(p);
    }

    const auto results = dedupMeasurements(coarse, fine);
    /* Measure the clock latency so that the latency measurements account only
     * for access latency. */
    double clockLatency = measureClock64Latency(numIters_);

    enc_["clock64_latency"] << clockLatency << " cycles " << std::endl;
    enc_["latency.csv"] << "bytes,avg_access_latency\n";
    for (const auto& [bytes, avgLatency] : fine)
      enc_["latency.csv"] << bytes << "," << avgLatency << "\n";
  }

 private:
  Encoder& enc_;
  uint64_t numExperiments_;
  uint64_t multiplier_;
  uint64_t numIters_;
  uint64_t startBytes_;
  uint64_t stride_; /* Stride in array elements, where each element is 8 bytes. */
  double threshCoarse_;
  uint64_t threshFine_;

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

  /** Run a single experiment, return the total number of device cycles. */
  static uint64_t runExperiment(uint64_t numIters, uint64_t stride, uint64_t numBytes) {
    if (numBytes % sizeof(uint64_t))
      throw std::runtime_error("number of bytes should be a multiple of 8 bytes");

    /* See the NOTE in the class header comment for an explanation as to why we
     * want stride to be coprime with the number of entries in the chain. */
    uint64_t numEntries = util::makeCoprime(numBytes / sizeof(uint64_t), stride);
    uint64_t* chain = initializeChain(numEntries, stride);

    uint64_t cycles;
    launchPChaseKernel(chain, numEntries * sizeof(uint64_t), numIters, &cycles);

    releaseChain(chain);
    return cycles;
  }

  static std::vector<std::pair<uint64_t, uint64_t>> findRidges(
      const std::vector<std::pair<uint64_t, double>>& measurements, const double thresh) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    for (uint64_t i = 0; i < measurements.size() - 1; i++)
      if (measurements[i + 1].second >= thresh * measurements[i].second)
        out.push_back({measurements[i].first, measurements[i + 1].first});
    return out;
  }

  const std::vector<std::pair<uint64_t, double>> runCoarse(uint64_t startBytes, uint64_t multiplier,
                                                           uint64_t numExperiments) const {
    std::vector<std::pair<uint64_t, double>> out;
    out.reserve(numExperiments_);
    uint64_t numBytes = startBytes;
    for (uint64_t i = 0; i < numExperiments; i++) {
      uint64_t cycles = runExperiment(numIters_, stride_, numBytes);
      double avgCyclesPerAccess = (double)cycles / (double)numIters_;
      out.push_back({numBytes, avgCyclesPerAccess});
      numBytes *= multiplier;
    }
    return out;
  }

  const std::vector<std::pair<uint64_t, double>> runFine(uint64_t lowerBound, uint64_t upperBound, uint64_t stride) {
    std::vector<std::pair<uint64_t, double>> out;
    out.reserve((upperBound - lowerBound + 1) / stride);
    for (uint64_t size = lowerBound; size <= upperBound; size += stride) {
      uint64_t cycles = runExperiment(numIters_, stride_, size);
      double avgCyclesPerAccess = (double)cycles / (double)numIters_;
      out.push_back({size, avgCyclesPerAccess});
    }
    return out;
  }

  static std::vector<std::pair<uint64_t, double>> dedupMeasurements(
      const std::vector<std::pair<uint64_t, double>>& coarse, const std::vector<std::pair<uint64_t, double>>& fine) {
    std::map<uint64_t, double> dedup;
    for (const auto& p : coarse)
      dedup.insert(p);

    for (const auto& [bytes, avgLatency] : fine) {
      auto [it, inserted] = dedup.insert({bytes, avgLatency});
      if (!inserted)
        it->second = (it->second + avgLatency) / 2.0;
    }
    return {dedup.begin(), dedup.end()};
  }
};

inline bool pchaseGPUBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<PChaseGPUBenchmark>(PChaseGPUBenchmark::benchmarkName);
  return true;
}();

#endif /* PCHASEGPUBENCHMARK_HH */
