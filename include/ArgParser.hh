#ifndef ARGPARSER_HH
#define ARGPARSER_HH

#include <cstdint>
#include <optional>
#include <sstream>
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
      using Elem = typename T::value_type;
      return parseList<Elem>(*val);
    } else {
      return parseValue<T>(*val);
    }
  }

 private:
  std::unordered_map<std::string, std::string> args_;

  /* Trait to detect std::vector<T> */
  template <typename T>
  struct is_vector : std::false_type {};
  template <typename T, typename A>
  struct is_vector<std::vector<T, A>> : std::true_type {};

  /* Utility function for parsing a single value */
  template <typename T>
  static T parseValue(const std::string& s) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint64_t>) {
      return static_cast<T>(std::stoull(s));
    } else if constexpr (std::is_same_v<T, double>) {
      return std::stod(s);
    } else {
      return s;  // string
    }
  }

  /* Utility function for parsing a list of values */
  template <typename T>
  static std::vector<T> parseList(const std::string& s) {
    std::vector<T> result;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, ',')) {
      if (!token.empty())
        result.push_back(parseValue<T>(token));
    }

    return result;
  }
};

};  // namespace benchmark

#endif /* ARGPARSER_HH */
