#ifndef SHAREDMEMBANDWIDTHBENCHMARK_HH
#define SHAREDMEMBANDWIDTHBENCHMARK_HH

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Util.hh>

/** 
 * Launch the shared memory bandwidth kernel.
 *
 * @param numElems Number of 32-bit words in shared memory, must be power of two.
 * @param numIters Iterations per thread of the measured loop.
 * @param threads Threads per block.
 * @param sharedBytes Size in bytes of the extern shared memory.
 * @param stride Stride in words used to create bank conflict patterns.
 * @param elapsedMsOut Returned elapsed milliseconds for measured kernel.
 */
void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, uint64_t sharedBytes,
                                    uint32_t stride, float* elapsedMsOut);

class SharedMemBandwidthBenchmark {
 public:
  static constexpr const char* benchmarkName = "shared_mem_bandwidth";

  SharedMemBandwidthBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    sizes_ = parser.getOr("sizes", std::vector<uint64_t>{4096, 8192, 16384, 32768, 49152});
    threads_ = parser.getOr("threads", std::vector<uint64_t>{32, 64, 128, 256});
    numIters_ = parser.getOr("num_iters", 10000UL);
    elemBytes_ = 4UL;
    reps_ = parser.getOr("reps", 3UL);
    strides_ = parser.getOr("strides", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
    warmupIters_ = parser.getOr("warmup_iters", 256UL);
  }

  std::string name() const { return benchmarkName; }

  void run() {
    enc_["shared_mem_bandwidth.csv"] << "bytes,threads,iters,reps,stride,timeMs,bandwidthGBps\n";

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
          std::vector<float> times;
          times.reserve(reps_);
          for (uint64_t r = 0; r < reps_; ++r) {
            float ms = 0.0f;
            launchSharedMemBandwidthKernel(numElems, static_cast<uint32_t>(numIters_), static_cast<uint32_t>(threads),
                                           bytes, static_cast<uint32_t>(stride), &ms);
            times.push_back(ms);
          }
          const float avgMs = std::accumulate(times.begin(), times.end(), 0.0f) / static_cast<float>(times.size());
          const double transfersPerIter = 2.0;
          const double bytesTransferred = static_cast<double>(elemBytes_) * transfersPerIter *
                                          static_cast<double>(numIters_) * static_cast<double>(threads);
          const double bandwidthGBps = (bytesTransferred / (avgMs / 1000.0)) / 1e9;

          enc_["shared_mem_bandwidth.csv"] << bytes << "," << threads << "," << numIters_ << "," << reps_ << ","
                                           << stride << "," << avgMs << "," << bandwidthGBps << "\n";
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
  uint64_t warmupIters_;
};

inline bool sharedMemBandwidthRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<SharedMemBandwidthBenchmark>(
      SharedMemBandwidthBenchmark::benchmarkName);
  return true;
}();

#endif
