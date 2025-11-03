/**
 * StridedAccessBenchmark - measure L1 cache bandwidth using strided loads
 *
 * This benchmark launches a configurable number of blocks/threads and performs
 * strided loads from per-block working sets sized to fit in L1. Timing is done
 * with CUDA events to compute aggregate bandwidth.
 */
#ifndef STRIDED_ACCESS_BENCHMARK_HH
#define STRIDED_ACCESS_BENCHMARK_HH

#include <cuda.h>
#include <cudaHelpers.cuh>
#include <stdexcept>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Encoder.hh>
#include <strided_access.hh>
#include <Common.hh>
#include <cuda_runtime.h>

class StridedAccessBenchmark {
 public:
  static constexpr const char* benchmarkName = "strided_access";

  StridedAccessBenchmark(Encoder& e, const std::vector<std::string>& args = {}) : enc_(e) {
    benchmark::ArgParser parser(args);
    workingSetSize_ = parser.getOr("working_set", 24 * common::KiB);
    stride_ = parser.getOr("stride", 1UL);
    iters_ = parser.getOr("iters", 1'000'000UL);
    threadsPerBlock_ = static_cast<int>(parser.getOr("threads_per_block", 256UL));
    numBlocks_ = static_cast<int>(parser.getOr("blocks", 0UL)); /* 0 => auto (SM count). */
  }

  std::string name() const { return benchmarkName; }

  void run() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t err;
    err = cudaGetDevice(&device);
    throwOnErr(err);
    err = cudaGetDeviceProperties(&prop, device);
    throwOnErr(err);

    int smCount = prop.multiProcessorCount;
    if (numBlocks_ <= 0)
      numBlocks_ = smCount; 
     

    /* Per-block working set in 4B elements. */
    const uint64_t elemsPerBlock = workingSetSize_ / sizeof(uint32_t);
    if (elemsPerBlock == 0)
      throw std::runtime_error("working_set must be >= 4 bytes");

    /* Allocate a contiguous buffer; each block uses a disjoint segment. */
    const uint64_t totalElems = elemsPerBlock * static_cast<uint64_t>(numBlocks_);
    std::vector<uint32_t> hostData(totalElems);
    for (uint64_t i = 0; i < totalElems; i++)
      hostData[i] = static_cast<uint32_t>(i * 2654435761u);

    uint64_t cycles = 0UL;
    launchStridedAccessKernel(hostData, stride_, iters_, threadsPerBlock_, numBlocks_, &cycles);

    /* TODO CHECK THIS CALCULATION */
    const uint64_t bytesRead = static_cast<uint64_t>(numBlocks_) * static_cast<uint64_t>(threadsPerBlock_) * iters_ *
                               sizeof(uint32_t);
    const double bytesPerCycle = static_cast<double>(bytesRead) / static_cast<double>(cycles);

    enc_["bandwidth.csv"] << "blocks,threads_per_block,working_set_B,stride_elems,iters,cycles,bytes,bytesPerCyclce\n";
    enc_["bandwidth.csv"] << numBlocks_ << "," << threadsPerBlock_ << "," << workingSetSize_ << "," << stride_ << ","
                           << iters_ << "," << cycles << "," << bytesRead << "," << bytesPerCycle << "\n";

    /* enc_["bandwidth.csv"] << "blocks,threads_per_block,working_set_B,stride_elems,iters,ms,bytes,bytesPerCyclce\n";
    enc_["bandwidth.csv"] << numBlocks_ << "," << threadsPerBlock_ << "," << workingSetSize_ << "," << stride_ << ","
                           << iters_ << "," << ms << "," << bytesRead << "," << bytesPerCycle << "\n";
    enc_.log() << "Strided L1 BW: " << (bytesPerCycle / (1024.0 * 1024.0 * 1024.0))
               << " GiB/s (" << bytesRead << " B over " << ms << " ms)\n"; */
  }

 private:
  Encoder& enc_;
  uint64_t workingSetSize_;
  uint64_t stride_;
  uint64_t iters_;
  int threadsPerBlock_;
  int numBlocks_;
};

inline bool stridedAccessBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<StridedAccessBenchmark>(
      StridedAccessBenchmark::benchmarkName);
  return true;
}();

#endif /* STRIDED_ACCESS_BENCHMARK_HH */
