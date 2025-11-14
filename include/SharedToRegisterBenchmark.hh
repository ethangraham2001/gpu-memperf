#ifndef SHARED_TO_REGISTER_BENCHMARK_HH
#define SHARED_TO_REGISTER_BENCHMARK_HH

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Util.hh>
#include <clock64.hh>

#include "shared_to_register_kernel.hh"

class SharedToRegisterBenchmark {
 public:
  static constexpr const char* benchmarkName = "shared_to_register";

  SharedToRegisterBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    sizes_ = parser.getOr("sizes", std::vector<uint64_t>{4096, 8192, 16384, 32768, 49152});
    threads_ = parser.getOr("threads", std::vector<uint64_t>{32, 64, 128, 256, 512});
    strides_ = parser.getOr("strides", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    elemBytes_ = 4UL;
    numIters_ = parser.getOr("num_iters", 10000UL);
    reps_ = parser.getOr("reps", 3UL);

    std::string modeStr = parser.getOr("mode", std::string(sharedToRegisterKernel::modeRead));
    mode_ = sharedToRegisterKernel::parseMode(modeStr);

    clockFreq_ = getMaxClockFrequencyHz();
  }

  std::string name() const { return benchmarkName; }

  void run() {
    enc_["shared_to_register.csv"] << "bytes,threads,stride,iters,cycles,bandwidthGBps\n";

    for (uint64_t bytes : sizes_) {
      uint32_t numElems = static_cast<uint32_t>(bytes / elemBytes_);

      for (uint64_t threads : threads_) {
        if (threads == 0)
          continue;
        if (threads > 1024)
          threads = 1024;

        for (uint64_t stride : strides_) {
          std::vector<uint64_t> cycles;
          cycles.reserve(reps_);

          for (uint64_t r = 0; r < reps_; ++r) {
            uint64_t cycle = 0;
            launchSharedToRegisterKernel(numElems, static_cast<uint32_t>(numIters_), static_cast<uint32_t>(threads),
                                         bytes, static_cast<uint32_t>(stride), mode_, &cycle);
            cycles.push_back(cycle);
          }

          const uint64_t avgCycles = std::accumulate(cycles.begin(), cycles.end(), 0) / (cycles.size());
          const double seconds = (double)avgCycles / (double)clockFreq_;
          const double transfersPerIter = (mode_ == sharedToRegisterKernel::READ_WRITE) ? 2.0 : 1.0;
          const double bytesTransferred = static_cast<double>(elemBytes_) * transfersPerIter *
                                          static_cast<double>(numIters_) * static_cast<double>(threads);
          double bandwidthGBps = (bytesTransferred / seconds) / 1e9;

          enc_["shared_to_register.csv"] << bytes << "," << threads << "," << stride << "," << numIters_ << ","
                                         << avgCycles << "," << bandwidthGBps << "\n";
        }
      }
    }
  }

 private:
  Encoder& enc_;
  std::vector<uint64_t> sizes_;
  std::vector<uint64_t> threads_;
  std::vector<uint64_t> strides_;
  uint64_t elemBytes_;
  uint64_t numIters_;
  uint64_t reps_;
  sharedToRegisterKernel::mode mode_;
  uint32_t clockFreq_;
};

inline bool sharedMemoryRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<SharedToRegisterBenchmark>(
      SharedToRegisterBenchmark::benchmarkName);
  return true;
}();

#endif