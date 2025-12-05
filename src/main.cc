#include <iostream>

#include <Benchmark.hh>
#include <GlobalToSharedBenchmark.hh>
#include <PchaseCPUBenchmark.hh>
#include <PchaseGPUBenchmark.hh>
#include <RandomAccessBenchmark.hh>
#include <SharedToRegisterBenchmark.hh>
#include <StridedAccessBenchmark.hh>
#include <Util.hh>

void __attribute__((noreturn)) usage() {
  std::cout << "Usage: ./gpu-memperf <benchmark-name> <parameters>" << std::endl;
  exit(0);
}

int main(int argc, char** argv) {
  if (argc < 2)
    usage();

  std::string name = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);

  Encoder encoder;
  try {
    /* Run the benchmark, serialize the results to std::cout. */
    benchmark::BenchmarkRegistry::instance().run(name, encoder, args);
    encoder.exportAll(name);
  } catch (std::out_of_range& e) {
    std::cerr << "Error: Unknown benchmark \"" << name << "\"\n";
    std::cerr << "Available benchmarks: ";
    const auto available = benchmark::BenchmarkRegistry::instance().availableBenchmarks();
    util::displayVector(std::cerr, available);
    return 1;
  } catch (std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
