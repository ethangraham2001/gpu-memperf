#ifndef GLOBAL_TO_SHARED_BENCHMARK_HH
#define GLOBAL_TO_SHARED_BENCHMARK_HH

#include <string>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>
#include <Common.hh>
#include <Encoder.hh>
#include <Types.hh>
#include <Util.hh>
#include <device_attributes.hh>
#include <global_to_shared.hh>

class GlobalToSharedBenchmark {
 public:
  static constexpr const char* benchmarkName = "global_to_shared";

  GlobalToSharedBenchmark(Encoder& e, const std::vector<std::string>& args) : enc_(e) {
    benchmark::ArgParser parser(args);
    numFlopsPerElem_ = parser.getOr<std::vector<uint64_t>>("flops_per_elem",
                                                           {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL, 512UL});
    threadsPerBlock_ = parser.getOr("threads_per_block", 1024UL);
    numBlocks_ = parser.getOr("num_blocks", 108UL);

    /* These aren't written out into the .csv file, so write them out into a
     * separate config file for posterity. */
    enc_["config"] << "threads_per_block: " << threadsPerBlock_ << "\n";
    enc_["config"] << "num_blocks: " << numBlocks_ << "\n";
  }

  void run() {
    const std::vector<globalToShared::mode> modes = {
        globalToShared::SYNC,
        globalToShared::ASYNC_2X_BUFFERED,
    };

    const std::string timeCsv = "time.csv";
    const std::string bandwidth = "bandwidth";

    enc_[timeCsv] << "mode,tile_size,buf_size,fops,ms\n";
    for (const auto mode : modes) {
      const std::string modeStr = globalToShared::modeStr(mode);
      enc_.log() << "Benching mode=" << modeStr << "\n";
      enc_[bandwidth] << "[" << modeStr << "]\n";

      for (const auto tileSize : {4 * common::KiB}) {
        enc_.log() << "tile size: " << tileSize << "\n";
        if (tileSize / sizeof(types::f32) < threadsPerBlock_)
          enc_.log() << "WARNING: tile is too small for all threads to work on it, which will likely result in idle "
                        "threads.\n";

        const uint32_t bufferSize = static_cast<uint32_t>(common::GiB);
        const auto numElems = util::countElements<uint32_t>(bufferSize);
        const std::vector<types::f32> globalBuffer = util::randomVector<types::f32>(numElems);

        enc_[bandwidth] << "\t[tile size = " << tileSize << "]\n";
        const auto launcher = getLauncher(tileSize);
        for (const auto flops : numFlopsPerElem_) {
          /* Dry-run to warm the GPU and cuda events context. */
          launcher(mode, globalBuffer, flops, threadsPerBlock_, numBlocks_);

          enc_.log() << "num flops: " << flops << "\n";
          enc_[bandwidth] << "\t\t[num flops = " << flops << "]\n";

          const float millis = launcher(mode, globalBuffer, flops, threadsPerBlock_, numBlocks_);
          const float bw = (float)bufferSize / (millis / 1000.0f);

          enc_[bandwidth] << "\t\t\tbandwidth: " << util::formatBytes(bw) << "/s\n";
          enc_[bandwidth] << "\t\t\tms       : " << millis << "\n\n";
          enc_[timeCsv] << modeStr << "," << tileSize << "," << bufferSize << "," << flops << "," << millis
                        << std::endl;
        }
      }
    }
  }

  std::string name() const { return benchmarkName; }

 private:
  Encoder& enc_;
  std::vector<uint64_t> numFlopsPerElem_;
  uint64_t threadsPerBlock_;
  uint64_t numBlocks_;

  using launcherFunc = float (*)(globalToShared::mode, const std::vector<types::f32>&, uint64_t, uint64_t, uint64_t);

  double approxBandwidth(uint64_t cycles, uint64_t clockFreq, uint64_t numBytes) {
    return (double)numBytes / (double)cycles * (double)clockFreq;
  }

  launcherFunc getLauncher(uint64_t tileSize) const {
    switch (tileSize) {
      /* Support only a small range of tileSize values. */
      case common::KiB:
        return launchGlobalToSharedKernel<common::KiB>;
      case 2 * common::KiB:
        return launchGlobalToSharedKernel<2 * common::KiB>;
      case 4 * common::KiB:
        return launchGlobalToSharedKernel<4 * common::KiB>;
      default:
        throw std::invalid_argument("tile size not supported");
    }
  }
};

inline bool globalToSharedBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<GlobalToSharedBenchmark>(
      GlobalToSharedBenchmark::benchmarkName);
  return true;
}();

#endif /* GLOBAL_TO_SHARED_BENCHMARK_HH */
