#ifndef PCHASECPUBENCHMARK_HH
#define PCHASECPUBENCHMARK_HH

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <ArgParser.hh>
#include <Benchmark.hh>

static constexpr uint64_t _cachelineSize = 64;
static constexpr uint64_t _paddingSize = 8;

struct Node {
  Node* next;
  uintptr_t pad[_paddingSize];
} __attribute__((aligned(_cachelineSize)));

/**
 * PchaseCPUBenchmark - use the PChase algorithm to estimate CPU cache sizes
 *
 * This serves as an example for how a benchmark should be defined. A benchmark
 * is described by the `Benchmark` concept.
 */
class PchaseCPUBenchmark {
 public:
  static constexpr const char* benchmarkName = "pchase_cpu";

  PchaseCPUBenchmark(const std::vector<std::string>& args = {}) {
    benchmark::ArgParser parser(args);
    numExperiments = parser.getOr("num_experiments", 12);
    multiplier = parser.getOr("multiplier", 2);
    numIters = parser.getOr("num_iters", 1e6);
    startBytes = parser.getOr("start_bytes", 1 << 16 /* 65536. */);
  }

  /* Implement the `Benchmark` concept. */
  std::string name() const { return benchmarkName; }
  void run(std::ostream& os) {
    const auto result = runBenchmarks(startBytes, multiplier, numExperiments, numIters);
    os << "bytes,nanos_per_access\n";
    for (const auto& [bytes, chasesPerNano] : result)
      os << bytes << "," << chasesPerNano << "\n";
  };

 private:
  uint64_t numExperiments;
  uint64_t multiplier;
  uint64_t numIters;
  uint64_t startBytes;

  static const std::vector<int> permutation(int n) {
    std::vector<int> out(n);
    std::iota(out.begin(), out.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(out.begin(), out.end(), gen);
    return out;
  }

  static const Node* generatePointerChain(int n) {
    const auto perm = permutation(n);
    Node* nodes = (Node*)malloc(n * sizeof(Node));
    for (auto i = 0; i < n - 1; i++)
      nodes[perm[i]].next = &nodes[perm[i + 1]];
    nodes[perm[n - 1]].next = &nodes[perm[0]];
    return nodes;
  }

  static uint64_t bytesToNumNodes(uint64_t bytes) {
    if (bytes % _cachelineSize)
      throw std::invalid_argument("bytes should be a multiple of cache line size");
    return bytes / _cachelineSize;
  }

  static void access(const Node* n, uint64_t numIters) {
    volatile uintptr_t sink;
    for (auto i = 0; i < numIters; i++)
      n = n->next;
    sink = (uintptr_t)n;
  }

  static uint64_t runExperiment(uint64_t numIters, uint64_t numNodes) {
    const Node* nodes = generatePointerChain(numNodes);

    access(nodes, numIters); /* Warm the cache. */

    const auto start = std::chrono::high_resolution_clock::now();
    access(nodes, numIters);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    free((void*)nodes);
    return dur.count();
  }

  static const std::vector<std::pair<uint64_t, uint64_t>> runBenchmarks(uint64_t startBytes, uint64_t multiplier,
                                                                        uint64_t numExperiments, uint64_t numIters) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    uint64_t numBytes = startBytes;
    for (int i = 0; i < numExperiments; i++) {
      uint64_t numNodes = bytesToNumNodes(numBytes);
      const auto nanos = runExperiment(numIters, numNodes);
      const auto nanosPerChase = nanos / numIters;
      out.push_back({numBytes, nanosPerChase});
      numBytes *= multiplier;
    }
    return out;
  }
};

/*
 * Registration lambda that is called before `main()`. This is required for the
 * benchmark to be made available in the registry. This lambda must be inline,
 * and has to be called here, guaranteeing that it is called before `main()`.
 */
inline bool pchaseBenchmarkRegistered = []() {
  benchmark::BenchmarkRegistry::instance().registerBenchmark<PchaseCPUBenchmark>(PchaseCPUBenchmark::benchmarkName);
  return true;
}(); /* NOTE: we call the function here with the `()`. */

#endif /* PCHASECPUBENCHMARK_HH */
