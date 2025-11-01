#include <iostream>
#include <string>

/* Register tests */
bool testArgParser();
bool testFailing();

int main() {
  bool allPassed = true;

  std::cout << "\n=== Running all tests ===\n";
  allPassed &= testArgParser();
  allPassed &= testFailing();

  return allPassed ? 0 : 1; /* Non-zero exit triggers CI failure */
}
