#include <iostream>
#include <string>
#include <vector>
#include "ArgParser.hh"
#include "TestHelpers.hh"

using namespace benchmark;

bool testArgParser() {
  std::cout << "[ArgParserTest] Testing...\n";
  bool passed = true;

  {
    ArgParser parser({});
    passed &= runTest("Default value", [&]() {
      int count = parser.getOr<int>("count", 10);
      return count == 10;
    });
    passed &= runTest("Default list value", [&]() {
      auto sizes = parser.getOr<std::vector<int>>("sizes", {256, 512});
      return (sizes == std::vector<int>{256, 512});
    });
  }
  {
    ArgParser parser({"--count=42"});
    passed &= runTest("Integer parsing", [&]() {
      int count = parser.getOr<int>("count", 0);
      return count == 42;
    });
  }
  {
    ArgParser parser({"--threshold=3.14"});
    passed &= runTest("Double parsing", [&]() {
      double threshold = parser.getOr<double>("threshold", 0.0);
      return threshold == 3.14;
    });
  }
  {
    ArgParser parser({"--name=gpu_benchmark"});
    passed &= runTest("String parsing", [&]() {
      std::string name = parser.getOr<std::string>("name", "default");
      return name == "gpu_benchmark";
    });
  }
  {
    ArgParser parser({"--sizes=4096,8192,16384"});
    passed &= runTest("Integer list parsing", [&]() {
      auto sizes = parser.getOr<std::vector<int>>("sizes", {});
      return (sizes == std::vector<int>{4096, 8192, 16384});
    });
  }
  {
    ArgParser parser({"--rates=1.1,2.2,3.3"});
    passed &= runTest("Double list parsing", [&]() {
      auto rates = parser.getOr<std::vector<double>>("rates", {});
      return (rates == std::vector<double>{1.1, 2.2, 3.3});
    });
  }

  std::cout << "[ArgParserTest] " << (passed ? "Passed" : "Failed") << "\n";
  return passed;
}
