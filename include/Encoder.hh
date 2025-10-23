#ifndef ENCODER_HH
#define ENCODER_HH

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

/**
 * LogHandle - logs data to both stdout and a string stream
 */
class LogHandle {
 public:
  LogHandle(std::stringstream& ss) : stream(ss) {}
  template <typename T>
  LogHandle& operator<<(const T& data) {
    stream << data;
    std::cout << data;
    return *this;
  }
  std::stringstream& stream;
};

/**
 * Encoder - multiplexes multiple outputs into a combined format
 *
 * The idea is to provide benchmarks with a means for writing their logs and
 * metrics out into a single file that contains a human readable section as
 * well as a machine-parsable portion of metrics.
 */
class Encoder {
 public:
  Encoder() : log_(files_[logFile]) {}

  std::stringstream& operator[](const std::string& filename) {
    if (filename == logFile)
      throw std::runtime_error(std::string(logFile) + " is reserved");
    return files_[filename];
  }

  void debugDumpAll(std::ostream& os) const {
    for (const auto& [filename, ss] : files_) {
      os << "### " << filename << " ###\n";
      os << ss.str() << "\n";
    }
  }

  LogHandle& log() {
    log_ << fmtTimestamp();
    return log_;
  }

  void exportAll(const std::string& benchName) const {
    const std::filesystem::path dirname = genFileName(benchName);
    std::filesystem::create_directories(dirname);
    for (const auto& [filename, ss] : files_) {
      std::ofstream out(dirname / filename);
      out << ss.str();
    }
    std::cout << fmtTimestamp() << " wrote results to " << dirname << std::endl;
  }

 private:
  static constexpr const char* logFile = "log.out";
  std::map<std::string, std::stringstream> files_;
  LogHandle log_;

  static std::string fmtTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[100];
    std::strftime(buf, sizeof(buf), "[%Y-%m-%d %H:%M:%S]", std::localtime(&time));
    std::stringstream ss;
    ss << buf << " ";
    return ss.str();
  }

  static std::string genFileName(const std::string& benchName) {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%H_%M_%S", std::localtime(&time));
    return benchName + "_" + buf;
  }
};

#endif /* ENCODER_HH */
