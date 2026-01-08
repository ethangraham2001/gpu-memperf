#ifndef RANDOM_ACCESS_BENCHMARK_HH
#define RANDOM_ACCESS_BENCHMARK_HH

#include <memory>
#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Common.hh>
#include <Encoder.hh>
#include <Types.hh>
#include <Util.hh>

#include <device_attributes.hh>
#include <random_access_kernel.hh>

/**
 * RandomAccessBenchmarkBase - required for templating
 *
 * Since we are doing dynamic dispatch on RandomAccessBenchmarks with different
 * data types, we need to have some abstract base class that implements the
 * required methods.
 */
class RandomAccessBenchmarkBase {
 public:
  virtual ~RandomAccessBenchmarkBase() = default;
  virtual void run() = 0;
  virtual std::string name() const = 0;
};

/**
 * RandomAccessBenchmarkGeneric - templated random access benchmark
 *
 * This supports using multiple data types (in particular, those defined in
 * Types.hh). This is interesting, as it allows us to measure random accesses
 * with various access granularities.
 */
template <typename DataType>
class RandomAccessBenchmarkGeneric : public RandomAccessBenchmarkBase {
 public:
  RandomAccessBenchmarkGeneric(Encoder& e, const std::vector<std::string>& args) : enc_(e) {
    benchmark::ArgParser parser(args);

    numWarps_ = parser.getOr<std::vector<uint64_t>>("num_warps", {1, 2, 4, 8, 16, 32});
    accessesPerThread_ = parser.getOr("num_accesses", (uint64_t)1e7);
    workingSetSize_ = parser.getOr("working_set", 8 * common::KiB);
    clockFreq_ = getMaxClockFrequencyHz();

    std::string modeStr = parser.getOr("mode", std::string("l1"));
    mode_ = randomAccessKernel::parseMode(modeStr);
    /* Pick a conservative value based on the cache level being benchmarked. */
    cacheSize_ = randomAccessKernel::defaultCacheSize(mode_);
    numBlocks_ = parser.getOr("num_blocks", randomAccessKernel::defaultNumBlocks(mode_));
    reps_ = parser.getOr("reps", (uint64_t)1);

    if (!configIsValid())
      throw std::invalid_argument("RandomAccessBenchmark config is invalid");
  }

  std::string name() const { return std::string("random_access_") + typeid(DataType).name(); }

  void run() {
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    const std::vector<uint32_t> indices = util::permutation<uint32_t>(numElems);
    std::vector<DataType> data(numElems);

    enc_["config"] << "clk_freq: " << clockFreq_ << "Hz\n";
    enc_["config"] << "access_granularity: " << sizeof(DataType) << "B\n";

    const std::string resultCSV = "result.csv";
    enc_[resultCSV] << "num_warps,cycles,bandwidth\n";

    for (uint64_t numWarps : numWarps_) {
      for (uint64_t i = 0; i < reps_; i++) {
        const auto res = runOne(numWarps);

        /* Write global result, per-thread results were removed when we started measuring per-block  */
        const double globalBandwidth = cyclesToBandwidth(res, numWarps);
        enc_[resultCSV] << numWarps << "," << res << "," << globalBandwidth << "\n";
        enc_.log() << numWarps << " warps, bandwidth: " << util::formatBytes(globalBandwidth) << "/S\n";
      }
    }
  }

 private:
  std::vector<uint64_t> numWarps_;
  randomAccessKernel::mode mode_;
  uint64_t accessesPerThread_;
  uint64_t workingSetSize_;
  uint64_t cacheSize_;
  uint32_t clockFreq_;
  uint64_t numBlocks_;
  uint64_t reps_;
  Encoder& enc_;

  uint64_t runOne(uint64_t numWarps) {
    const uint64_t numThreads = common::threadsPerWarp * numWarps;
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    const std::vector<uint32_t> indices = util::permutation<uint32_t>(numElems);
    std::vector<DataType> data(numElems);
    return launchRandomAccessKernel(data, indices, accessesPerThread_, numThreads, numBlocks_, mode_);
  }

  double cyclesToBandwidth(uint64_t cycles, uint64_t numWarps) {
    uint64_t bytesAccessed = numBlocks_ * numWarps * common::threadsPerWarp * accessesPerThread_ * sizeof(DataType);
    double seconds = (double)cycles / (double)clockFreq_;
    return (double)bytesAccessed / seconds;
  }

  uint64_t totalBytesAccessed(uint64_t numWarps) {
    return common::threadsPerWarp * numWarps * accessesPerThread_ * sizeof(DataType);
  }

  bool configIsValid() {
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    uint64_t indicesSize = numElems * sizeof(uint32_t);
    if (workingSetSize_ + indicesSize > cacheSize_)
      return false;
    /* TODO: once we support passing the number of warps as a list, we should
     * be adding a validation step for that too (e.g, in the L1 mode, we
     * shouldn't have more warps than can fit in a block. */
    return true;
  }
};

/**
 * RandomAccessBenchmark - dynamic dispatcher for templated benchmarks
 *
 * Implements dynamic dispatch for templated random access benchmarks with a set
 * of supported types (f8 through to f64).
 */
class RandomAccessBenchmark : public RandomAccessBenchmarkBase {
 public:
  static constexpr const char* benchmarkName = "random_access";

  RandomAccessBenchmark(Encoder& e, const std::vector<std::string>& args) {
    benchmark::ArgParser parser(args);
    std::string dataType = parser.getOr("data_type", std::string("f32"));

    if (dataType == "f32")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f32>>(e, args);
    else if (dataType == "f64")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f64>>(e, args);
    else
      throw std::runtime_error("unknown type: " + dataType);
    dataType_ = dataType;
  }

  void run() { bench_->run(); }

  std::string name() const { return bench_->name(); };

 private:
  std::string dataType_;
  std::unique_ptr<RandomAccessBenchmarkBase> bench_;
};

inline bool randomAccessBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<RandomAccessBenchmark>(
      RandomAccessBenchmark::benchmarkName);
  return true;
}();

#endif /* RANDOM_ACCESS_BENCHMARK_HH */
