#ifndef ARGPARSER_HH
#define ARGPARSER_HH

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace benchmark {

/**
 * ArgParser - parses command line arguments
 *
 * The ArgParser is a helper class for parsing a list of named arguments of
 * the form "--arg1=value1, --arg2=value2, ... --argN=valueN". It supports
 * arguments of the following types:
 *
 * - Integers (`int`, `uint64_t`).
 * - Doubles (`double`).
 * - Strings (`const char*`, `std::string`).
 * - Lists (`std::vector<T>` where T is a generic of the above types).
 */
class ArgParser {
 public:
  ArgParser(const std::vector<std::string>& args) {
    for (const auto& arg : args) {
      auto pos = arg.find('=');
      if (pos != std::string::npos) {
        std::string key = arg.substr(0, pos);
        std::string value = arg.substr(pos + 1);

        // Strip leading dashes
        while (!key.empty() && key[0] == '-') {
          key = key.substr(1);
        }

        args_[key] = value;
      }
    }
  }

  std::optional<std::string> get(const std::string& key) const {
    auto it = args_.find(key);
    return it != args_.end() ? std::optional{it->second} : std::nullopt;
  }

  template <typename T>
  T getOr(const std::string& key, T defaultValue) const {
    auto val = get(key);
    if (!val)
      return defaultValue;

    if constexpr (is_vector<T>::value) {
      using ElemType = typename T::value_type;
      return parseList<ElemType>(*val);
    } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint64_t>) {
      return std::stoull(*val);
    } else if constexpr (std::is_same_v<T, double>) {
      return std::stod(*val);
    } else {
      return *val;  // string
    }
  }

 private:
  std::unordered_map<std::string, std::string> args_;

  /* Generic list parser */
  template <typename T>
  struct is_vector : std::false_type {};
  template <typename T, typename A>
  struct is_vector<std::vector<T, A>> : std::true_type {};

  template <typename T>
  static std::vector<T> parseList(const std::string& s) {
    std::vector<T> result;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, ',')) {
      if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint64_t>) {
        result.push_back(std::stoull(token));
      } else if constexpr (std::is_same_v<T, double>) {
        result.push_back(std::stod(token));
      } else {
        result.push_back(token);
      }
    }
    return result;
  }
};

};  // namespace benchmark

#endif /* ARGPARSER_HH */
