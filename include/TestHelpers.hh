#include <exception>
#include <functional>
#include <iostream>
#include <string>

/* ANSI color codes for terminal output */
constexpr const char* GREEN = "\033[32m";
constexpr const char* RED = "\033[31m";
constexpr const char* RESET = "\033[0m";

/* Test runner function */
bool runTest(const std::string& testName, const std::function<bool()>& func) {
  bool success = false;
  std::string error_message;

  try {
    success = func();
    if (!success)
      error_message = "assertion error";
  } catch (const std::runtime_error& e) {
    success = false;
    error_message = std::string("std::runtime_error: ") + e.what();
  } catch (const std::exception& e) {
    success = false;
    error_message = std::string("std::exception: ") + e.what();
  }

  if (success)
    std::cout << GREEN << "[PASSED]" << RESET << " " << testName << '\n';
  else
    std::cout << RED << "[FAILED]" << RESET << " " << testName << ": " << error_message << '\n';

  return success;
}