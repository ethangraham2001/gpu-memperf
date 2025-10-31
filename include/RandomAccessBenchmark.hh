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
#include <clock64.hh>
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
    accessesPerThread_ = parser.getOr("num_accesses", 10000000UL);
    workingSetSize_ = parser.getOr("working_set", 1UL << 13 /* 8KiB. */);
    clockFreq_ = getClockFrequencyHz();

    std::string modeStr = parser.getOr("mode", std::string("l1"));
    mode_ = randomAccessKernel::parseMode(modeStr);
    /* Pick a conservative value based on the cache level being benchmarked. */
    cacheSize_ = randomAccessKernel::defaultCacheSize(mode_);
    /* XXX: assumes that we don't want to parameterize the number of blocks. */
    numBlocks_ = randomAccessKernel::defaultNumBlocks(mode_);

    if (!configIsValid())
      throw std::invalid_argument("RandomAccessBenchmark config is invalid");
  }

  std::string name() const { return std::string("random_access_") + typeid(DataType).name(); }

  void run() {
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    const std::vector<uint32_t> indices = util::permutation<uint32_t>(numElems);
    std::vector<DataType> data(numElems);

    enc_["dev_clk_freq"] << clockFreq_ << std::endl;
    enc_["access_granularity"] << sizeof(DataType) << "B" << std::endl;

    const std::string globalCSV = "global_bw.csv";
    const std::string perThreadCSV = "per_thread_bw.csv";
    enc_[globalCSV] << "num_warps,cycles,bandwidth\n";
    enc_[perThreadCSV] << "num_warps,tid,cycles\n";

    for (uint64_t numWarps : numWarps_) {
      const auto res = runOne(numWarps);

      /* Write global result. */
      const double globalBandwidth = cyclesToBandwidth(res.first, numWarps);
      enc_[globalCSV] << numWarps << "," << res.first << "," << globalBandwidth << "\n";
      enc_.log() << numWarps << " warps, bandwidth: " << util::formatBytes(globalBandwidth) << "/S\n";

      /* Write per-thread results. */
      for (uint64_t tid = 0; tid < res.second.size(); tid++) {
        const uint64_t threadResult = res.second[tid];
        enc_[perThreadCSV] << numWarps << "," << tid << "," << threadResult << "\n";
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
  Encoder& enc_;

  std::pair<uint64_t, std::vector<uint64_t>> runOne(uint64_t numWarps) {
    const uint64_t numThreads = common::threadsPerWarp * numWarps;
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    const std::vector<uint32_t> indices = util::permutation<uint32_t>(numElems);
    std::vector<DataType> data(numElems);
    return launchRandomAccessKernel(data, indices, accessesPerThread_, numThreads, numBlocks_, mode_);
  }

  double cyclesToBandwidth(uint64_t cycles, uint64_t numWarps) {
    uint64_t bytesAccessed = numWarps * common::threadsPerWarp * accessesPerThread_ * sizeof(DataType);
    double seconds = (double)cycles / (double)clockFreq_;
    return (double)bytesAccessed / seconds;
  }

  uint64_t totalBytesAccessed(uint64_t numWarps) {
    return common::threadsPerWarp * numWarps * accessesPerThread_ * sizeof(DataType);
  }

  bool configIsValid() {
    uint64_t numElems = workingSetSize_ / sizeof(DataType);
    uint64_t indicesSize = numElems * sizeof(uint32_t);
    if (workingSetSize_ + indicesSize >= cacheSize_)
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
class RandomAccessBenchmark {
 public:
  static constexpr const char* benchmarkName = "random_access";

  RandomAccessBenchmark(Encoder& e, const std::vector<std::string>& args) {
    benchmark::ArgParser parser(args);
    std::string dataType = parser.getOr("data_type", std::string("int32"));

    if (dataType == "f8")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f8>>(e, args);
    else if (dataType == "f16")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f16>>(e, args);
    else if (dataType == "f32")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f32>>(e, args);
    else if (dataType == "f64")
      bench_ = std::make_unique<RandomAccessBenchmarkGeneric<types::f64>>(e, args);
    else
      throw std::runtime_error("unknown type: " + dataType);
    dataType_ = dataType;
  }

  void run() { bench_->run(); }

  std::string name() { return bench_->name(); };

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
