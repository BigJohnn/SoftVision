#pragma once
#include <string>
#include <sstream>

namespace utils{
    std::string GetFileExtension(std::string const& filepath)
    {
        return filepath.substr(filepath.find_last_of(".") + 1);
    }

    void split(std::vector<std::string> &out_words, std::string const& str, char const& pivot)
    {
      std::istringstream f(str);
      std::string s;
      while (getline(f, s, pivot)) {
          out_words.push_back(s);
      }
    }
}
