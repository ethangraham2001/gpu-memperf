#ifndef SHARED_TO_REGISTER_BENCHMARK_HH
#define SHARED_TO_REGISTER_BENCHMARK_HH

#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Types.hh>
#include <Util.hh>
#include <device_attributes.hh>

#include <shared_to_register_kernel.hh>

class SharedToRegisterBenchmark {
 public:
  static constexpr const char* benchmarkName = "shared_to_register";

  SharedToRegisterBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    sizes_ = parser.getOr("sizes", std::vector<uint64_t>{4096, 8192, 16384, 32768});
    threads_ = parser.getOr("threads", std::vector<uint64_t>{32, 64, 128, 256, 512});
    strides_ = parser.getOr("strides", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    numIters_ = parser.getOr("num_iters", 100000UL);
    reps_ = parser.getOr("reps", 3UL);
    numBlocks_ = static_cast<int>(parser.getOr("blocks", 0UL)); /* 0 => auto (SM count). */

    std::string modeStr = parser.getOr("mode", std::string(sharedToRegister::modeRead));
    mode_ = sharedToRegister::parseMode(modeStr);
  }

  std::string name() const { return benchmarkName; }

  void run() {
    if (numBlocks_ <= 0)
      numBlocks_ = getSmCount(0);

    const std::string resultCsv = "result.csv";
    enc_[resultCsv] << "run,bytes,threads,stride,iters,timeMs,bandwidthGBps\n";
    enc_.log() << benchmarkName << ": mode=" << mode_ << "\n";
    enc_.log() << benchmarkName << ": numBlocks=" << numBlocks_ << "\n";

    for (uint64_t bytes : sizes_) {
      uint32_t numElems = static_cast<uint32_t>(bytes / sizeof(types::f64));

      for (uint64_t threads : threads_) {
        if (threads == 0)
          continue;
        if (threads > 1024)
          threads = 1024;

        for (uint64_t r = 0; r < reps_; ++r) {
          for (uint64_t stride : strides_) {
            float ms = 0.0f;
            launchSharedToRegisterKernel(numElems, static_cast<uint32_t>(numIters_), static_cast<uint32_t>(threads),
                                         static_cast<uint32_t>(numBlocks_), bytes, static_cast<uint32_t>(stride), mode_,
                                         &ms);

            const double transfersPerIter = (mode_ == sharedToRegister::READ_WRITE) ? 2.0 : 1.0;
            const double bytesTransferred = static_cast<double>(sizeof(types::f64)) * transfersPerIter *
                                            static_cast<double>(numIters_) * static_cast<double>(numBlocks_) *
                                            static_cast<double>(threads);
            double bandwidthGBps = (bytesTransferred / (ms / 1000.0f)) / 1e9;

            enc_[resultCsv] << r + 1 << "," << bytes << "," << threads << "," << stride << "," << numIters_ << "," << ms
                            << "," << bandwidthGBps << "\n";
          }
        }
      }
    }
  }

 private:
  Encoder& enc_;
  std::vector<uint64_t> sizes_;
  std::vector<uint64_t> threads_;
  std::vector<uint64_t> strides_;
  uint64_t numIters_;
  uint64_t reps_;
  int numBlocks_;
  sharedToRegister::mode mode_;
};

inline bool sharedMemoryRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<SharedToRegisterBenchmark>(
      SharedToRegisterBenchmark::benchmarkName);
  return true;
}();

#endif
