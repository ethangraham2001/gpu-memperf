#include <iostream>
#include <string>

/* Register tests */
bool testArgParser();

int main() {
  bool allPassed = true;

  std::cout << "\n=== Running all tests ===\n";
  allPassed &= testArgParser();

  return allPassed ? 0 : 1; /* Non-zero exit triggers CI failure */
}
