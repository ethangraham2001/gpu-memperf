/**
 * StridedAccessBenchmark - measure L1 cache bandwidth using strided loads
 *
 * This benchmark launches a configurable number of blocks/threads and performs
 * strided loads from per-block working sets sized to fit in L1. Timing is done
 * with CUDA events to compute aggregate bandwidth.
 */
#ifndef STRIDED_ACCESS_BENCHMARK_HH
#define STRIDED_ACCESS_BENCHMARK_HH

#include <Types.hh>
#include <device_attributes.hh>
#include <stdexcept>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <CachePolicy.hh>
#include <Common.hh>
#include <Encoder.hh>
#include <Util.hh>
#include <strided_access.hh>

/**
 * StridedAccessBenchmarkBase - required for templating
 *
 * Since we are doing dynamic dispatch on StridedAccessBenchmarks with different
 * data types, we need to have some abstract base class that implements the
 * required methods.
 */
class StridedAccessBenchmarkBase {
 public:
  virtual ~StridedAccessBenchmarkBase() = default;
  virtual void run() = 0;
  virtual std::string name() const = 0;
};

template <typename DataType>
class StridedAccessBenchmarkGeneric : public StridedAccessBenchmarkBase {
 public:
  StridedAccessBenchmarkGeneric(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    mode = parser.getOr("mode", std::string("L1"));
    strides_ = parser.getOr<std::vector<uint64_t>>("stride", {1, 2, 4, 8, 16, 32});
    iters_ = parser.getOr("iters", 1000000UL);
    threadsPerBlock_ = static_cast<int>(parser.getOr("threads_per_block", common::maxThreadsPerBlock));
    numBlocks_ = static_cast<int>(parser.getOr("blocks", 0UL)); /* 0 => auto (SM count). */

    /* TODO: Refactor with RandomAccessBenchmark */
    if (mode == "L1") {
      workingSetSize_ = 100 * common::KiB; /* L1. */
      cachePolicy_ = cacheload::CachePolicy::L1;
    } else if (mode == "L2") {
      workingSetSize_ = 25 * common::MiB; /* L2. */
      cachePolicy_ = cacheload::CachePolicy::L2;
    } else if (mode == "DRAM") {
      workingSetSize_ = 2 * common::GiB; /* DRAM. */
      cachePolicy_ = cacheload::CachePolicy::DRAM;
    } else {
      throw std::runtime_error("Invalid mode, please select one of {L1, L2, DRAM}");
    }
  }

  std::string name() const { return std::string("strided_access") + typeid(DataType).name(); }

  void run() {
    if (numBlocks_ <= 0)
      numBlocks_ = getSmCount(0);

    const uint64_t numElems = util::bytesToNumElems<DataType>(workingSetSize_);
    if (!numElems) {
      throw std::runtime_error("working_set too small");
    }

    std::vector<DataType> hostData(numElems);

    uint32_t clockFreq_ = getMaxClockFrequencyHz();
    enc_["bandwidth.csv"] << "blocks,threads_per_block,working_set,iters,stride,bandwidth\n";

    for (uint64_t stride_ : strides_) {

      uint64_t cycles = 0UL;
      launchStridedAccessKernel(hostData, cachePolicy_, stride_, iters_, threadsPerBlock_, numBlocks_, &cycles);
      const uint64_t bytesRead =
          static_cast<uint64_t>(numBlocks_) * static_cast<uint64_t>(threadsPerBlock_) * iters_ * sizeof(DataType);

      double seconds = (double)cycles / (double)clockFreq_;
      double bandwidth = (double)bytesRead / (double)seconds;

      enc_["bandwidth.csv"] << numBlocks_ << "," << threadsPerBlock_ << ","
                            << util::formatBytes((double)workingSetSize_) << "," << iters_ << "," << stride_ << ","
                            << bandwidth << "\n";

      enc_.log() << "Memory: " << mode << " Stride: " << stride_ << " BW:" << util::formatBytes(bandwidth) << "\n";
    }
  }

 private:
  Encoder& enc_;
  uint64_t workingSetSize_;
  std::vector<uint64_t> strides_;
  uint64_t iters_;
  int threadsPerBlock_;
  int numBlocks_;
  std::string mode;
  cacheload::CachePolicy cachePolicy_{cacheload::CachePolicy::L1};
};

/**
 * StridedAccessBenchmark - dynamic dispatcher for templated benchmarks
 *
 * Implements dynamic dispatch for templated strided access benchmarks with a set
 * of supported types (f8 through to f64).
 */
class StridedAccessBenchmark : public StridedAccessBenchmarkBase {
 public:
  static constexpr const char* benchmarkName = "strided_access";

  StridedAccessBenchmark(Encoder& e, const std::vector<std::string>& args) {
    benchmark::ArgParser parser(args);
    dataType_ = parser.getOr("data_type", std::string("int32"));

    if (dataType_ == "f32")
      bench_ = std::make_unique<StridedAccessBenchmarkGeneric<types::f32>>(e, args);
    else if (dataType_ == "f64")
      bench_ = std::make_unique<StridedAccessBenchmarkGeneric<types::f64>>(e, args);
    else
      throw std::runtime_error("unknown type: " + dataType_);
  }

  void run() { bench_->run(); }

  std::string name() const { return bench_->name(); };

 private:
  std::string dataType_;
  std::unique_ptr<StridedAccessBenchmarkBase> bench_;
};

inline bool stridedAccessBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<StridedAccessBenchmark>(
      StridedAccessBenchmark::benchmarkName);
  return true;
}();

#endif /* STRIDED_ACCESS_BENCHMARK_HH */
