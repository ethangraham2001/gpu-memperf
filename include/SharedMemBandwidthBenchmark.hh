#ifndef SHAREDMEMBANDWIDTHBENCHMARK_HH
#define SHAREDMEMBANDWIDTHBENCHMARK_HH

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Util.hh>
#include <clock64.hh>

/**
 * Launch the shared memory bandwidth kernel.
 *
 * @param numElems Number of 32-bit words in shared memory, must be power of two.
 * @param numIters Iterations per thread of the measured loop.
 * @param threads Threads per block.
 * @param sharedBytes Size in bytes of the extern shared memory.
 * @param stride Stride in words used to create bank conflict patterns.
 * @param mode 0=read, 1=write, 2=read+write.
 * @param cycle Returned cycles for measured kernel.
 */
void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                    uint32_t stride, uint32_t mode, uint64_t* cycle);

class SharedMemBandwidthBenchmark {
 public:
  static constexpr const char* benchmarkName = "shared_mem_bandwidth";

  SharedMemBandwidthBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    sizes_ = parser.getOr("sizes", std::vector<uint64_t>{4096, 8192, 16384, 32768, 49152});
    threads_ = parser.getOr("threads", std::vector<uint64_t>{32, 64, 128, 256});
    strides_ = parser.getOr("strides", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    elemBytes_ = 4UL;
    numIters_ = parser.getOr("num_iters", 10000UL);
    reps_ = parser.getOr("reps", 3UL);
    mode_ = parser.getOr("mode", 0UL); /* 0=read, 1=write, 2=read+write. */
    clockFreq_ = getMaxClockFrequencyHz();
  }

  std::string name() const { return benchmarkName; }

  void run() {
    enc_["shared_mem_bandwidth.csv"] << "bytes,threads,stride,iters,cycles,bandwidthGBps\n";

    for (uint64_t bytes : sizes_) {
      const uint64_t numElems64 = bytes / elemBytes_;
      if (numElems64 > std::numeric_limits<uint32_t>::max()) {
        enc_.log() << "Skipping size " << bytes << " (element count exceeds 32-bit limit).\n";
        continue;
      }
      const uint32_t numElems = static_cast<uint32_t>(numElems64);

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
            launchSharedMemBandwidthKernel(numElems, static_cast<uint32_t>(numIters_), static_cast<uint32_t>(threads),
                                           bytes, static_cast<uint32_t>(stride), static_cast<uint32_t>(mode_), &cycle);
            cycles.push_back(cycle);
          }

          const uint64_t avgCycles =
              std::accumulate(cycles.begin(), cycles.end(), 0) / static_cast<uint64_t>(cycles.size());
          const double seconds = (double)avgCycles / (double)clockFreq_;
          const double transfersPerIter = (mode_ == 2) ? 2.0 : 1.0; /* mode 2: read+write */
          const double bytesTransferred = static_cast<double>(elemBytes_) * transfersPerIter *
                                          static_cast<double>(numIters_) * static_cast<double>(threads);
          const double bandwidthGBps = (bytesTransferred / seconds) / 1e9;

          enc_["shared_mem_bandwidth.csv"] << bytes << "," << threads << "," << stride << "," << numIters_ << ","
                                           << avgCycles << "," << bandwidthGBps << std::fixed << "\n";
        }
      }
    }
  }

 private:
  Encoder& enc_;
  std::vector<uint64_t> sizes_;
  std::vector<uint64_t> threads_;
  uint64_t numIters_;
  uint64_t elemBytes_;
  uint64_t reps_;
  std::vector<uint64_t> strides_;
  uint64_t mode_;
  uint32_t clockFreq_;
};

inline bool sharedMemBandwidthRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<SharedMemBandwidthBenchmark>(
      SharedMemBandwidthBenchmark::benchmarkName);
  return true;
}();

#endif
