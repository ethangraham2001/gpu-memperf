#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include "ArgParser.hh"

using namespace benchmark;

template <typename Func>
void runTest(const std::string& testName, Func func) {
  bool success = false;
  std::string error_message;

  try {
    success = func();
    if (!success)
      error_message = "test failed";
  } catch (const std::exception& e) {
    success = false;
    error_message = std::string("runtime exception: ") + e.what();
  } catch (...) {
    success = false;
    error_message = "unknown runtime error";
  }

  if (success) {
    std::cout << testName << " ✅\n";
  } else {
    std::cout << testName << " ❌: " << error_message << "\n";
  }
}

int main() {
  std::cout << "[ArgParserTest] Testing...\n";
  {
    ArgParser parser({});
    runTest("Default value", [&parser]() {
      int count = parser.getOr<int>("count", 10);
      return count == 10;
    });
    runTest("Default list value", [&parser]() {
      auto sizes = parser.getOr<std::vector<int>>("sizes", {256, 512});
      return (sizes == std::vector<int>{256, 512});
    });
  }
  {
    ArgParser parser({"--count=42"});
    runTest("Integer parsing", [&parser]() {
      int count = parser.getOr<int>("count", 0);
      return count == 42;
    });
  }
  {
    ArgParser parser({"--threshold=3.14"});
    runTest("Double parsing", [&parser]() {
      double threshold = parser.getOr<double>("threshold", 0.0);
      return threshold == 3.14;
    });
  }
  {
    ArgParser parser({"--name=gpu_benchmark"});
    runTest("String parsing", [&parser]() {
      std::string name = parser.getOr<std::string>("name", "default");
      return name == "gpu_benchmark";
    });
  }
  {
    ArgParser parser({"--sizes=4096,8192,16384"});
    runTest("Integer list parsing", [&parser]() {
      auto sizes = parser.getOr<std::vector<int>>("sizes", {});
      return (sizes == std::vector<int>{4096, 8192, 16384});
    });
  }

  {
    ArgParser parser({"--rates=1.1,2.2,3.3"});
    runTest("Double list parsing", [&parser]() {
      auto rates = parser.getOr<std::vector<double>>("rates", {});
      return (rates == std::vector<double>{1.1, 2.2, 3.3});
    });
  }

  std::cout << "[ArgParserTest] Completed.\n";
  return 0;
}
