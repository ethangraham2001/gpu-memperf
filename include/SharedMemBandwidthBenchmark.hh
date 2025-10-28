#ifndef SHAREDMEMBANDWIDTHBENCHMARK_HH
#define SHAREDMEMBANDWIDTHBENCHMARK_HH

#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Util.hh>
#include "cudaHelpers.cuh"

void launchSharedMemBandwidthKernel(uint32_t numElems, uint32_t numIters, uint32_t threads, size_t sharedBytes,
                                    float* elapsedMsOut);

class SharedMemBandwidthBenchmark {
 public:
  static constexpr const char* benchmarkName = "shared_mem_bandwidth";

  SharedMemBandwidthBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    sizes_ = parseList(parser.getOr("sizes", std::string("4096,8192,16384,32768,49152")));
    threads_ = parseList(parser.getOr("threads", std::string("32,64,128,256")));
    numIters_ = parser.getOr("num_iters", 10000UL);
    elemBytes_ = parser.getOr("elem_bytes", 4UL);
    reps_ = parser.getOr("reps", 3UL);
  }

  std::string name() const { return benchmarkName; }

  void run() {
    enc_["shared_mem_bandwidth.csv"] << "bytes,threads,iters,reps,time_ms,bandwidth_gbps\n";

    int device;
    throwOnErr(cudaGetDevice(&device));
    int maxSharedKB = 0;
    throwOnErr(cudaDeviceGetAttribute(&maxSharedKB, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (maxSharedKB == 0)
      throwOnErr(cudaDeviceGetAttribute(&maxSharedKB, cudaDevAttrMaxSharedMemoryPerBlock, device));
    const size_t maxSharedBytes = static_cast<size_t>(maxSharedKB);

    for (size_t bytes : sizes_) {
      if (bytes > maxSharedBytes) {
        enc_.log() << "Skipping size " << bytes << " (exceeds device shared memory limit)\n";
        continue;
      }

      const size_t numElems64 = bytes / elemBytes_;
      if (numElems64 > std::numeric_limits<uint32_t>::max()) {
        enc_.log() << "Skipping size " << bytes << " (element count exceeds 32-bit limit)\n";
        continue;
      }
      const uint32_t numElems = static_cast<uint32_t>(numElems64);

      for (size_t threads : threads_) {
        if (threads == 0)
          continue;
        if (threads > 1024)
          threads = 1024;

        std::vector<float> times;
        times.reserve(reps_);
        for (size_t r = 0; r < reps_; ++r) {
          float ms = 0.0f;
          launchSharedMemBandwidthKernel(numElems, static_cast<uint32_t>(numIters_), static_cast<uint32_t>(threads),
                                         bytes, &ms);
          times.push_back(ms);
        }

        const float avgMs = std::accumulate(times.begin(), times.end(), 0.0f) / static_cast<float>(times.size());

        const double bytesTransferred =
            static_cast<double>(elemBytes_) * 2.0 * static_cast<double>(numIters_) * static_cast<double>(threads);
        const double bandwidthGbps = (bytesTransferred / (avgMs / 1000.0)) / 1e9;

        enc_["shared_mem_bandwidth.csv"] << bytes << "," << threads << "," << numIters_ << "," << reps_ << "," << avgMs
                                         << "," << bandwidthGbps << "\n";
      }
    }
  }

 private:
  Encoder& enc_;
  std::vector<size_t> sizes_;
  std::vector<size_t> threads_;
  uint64_t numIters_;
  size_t elemBytes_;
  size_t reps_;

  static std::vector<size_t> parseList(const std::string& s) {
    std::vector<size_t> out;
    std::stringstream ss(s);
    for (std::string token; std::getline(ss, token, ',');)
      if (!token.empty())
        out.push_back(std::stoull(token));
    if (out.empty())
      out.push_back(4096);
    return out;
  }
};

inline bool sharedMemBandwidthRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<SharedMemBandwidthBenchmark>(
      SharedMemBandwidthBenchmark::benchmarkName);
  return true;
}();

#endif
