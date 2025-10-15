#include <iostream>

#include <Benchmark.hh>
#include <PchaseCPUBenchmark.hh>
#include <Util.hh>

int main(int argc, char** argv) {
  if (argc < 2)
    throw std::invalid_argument("insufficient arguments");

  std::string name = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);

  try {
    /* Run the benchmark, serialize the results to std::cout. */
    benchmark::BenchmarkRegistry::instance().run(name, args, std::cout);
  } catch (std::out_of_range& e) {
    std::cerr << "Error: Unknown benchmark \"" << name << "\"\n";
    std::cerr << "Available benchmarks: ";
    const auto available = benchmark::BenchmarkRegistry::instance().availableBenchmarks();
    util::displayVector(std::cerr, available);
  } catch (std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
  }
}
