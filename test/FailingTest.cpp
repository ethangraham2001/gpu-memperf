#include <iostream>
#include <string>
#include "TestHelpers.hh"

bool testFailing() {
  std::cout << "[FailingTest] Testing...\n";
  bool passed = true;

  passed &= runTest("Should fail CI", [&]() { return false; });

  std::cout << "[FailingTest] " << (passed ? "Passed" : "Failed") << "\n";
  return passed;
}
