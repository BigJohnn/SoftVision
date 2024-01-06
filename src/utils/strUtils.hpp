#pragma once
#include <string>
#include <sstream>

namespace utils{

    std::string GetFileExtension(std::string const& filepath)
    {
        return filepath.substr(filepath.find_last_of("."));
    }

    std::string GetFileName(std::string const& filepath)
    {
        return filepath.substr(filepath.find_last_of("/") + 1);
    }

    std::string GetFileNameStem(std::string const& filepath)
    {
        auto&& last_slide_pos = filepath.find_last_of("/");
        return filepath.substr(last_slide_pos + 1, filepath.find_last_of(".") - last_slide_pos - 1);
    }

    std::string GetParentPath(std::string const& filepath)
    {
        return filepath.substr(0, filepath.find_last_of("/"));
    }

    void split(std::vector<std::string> &out_words, std::string const& str, char const& pivot)
    {
      std::istringstream f(str);
      std::string s;
      while (getline(f, s, pivot)) {
          out_words.push_back(s);
      }
    }

    std::string to_lower_copy(std::string const& str)
    {
        std::string ret = str;
        std::transform(str.begin(), str.end(), ret.begin(), ::tolower);
        return ret;
    }

    std::string& temp_directory_path();

    std::string random_string( size_t length = 10 )
    {
        auto randchar = []() -> char
        {
            const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(charset) - 1);
            return charset[ rand() % max_index ];
        };
        std::string str(length,0);
        std::generate_n( str.begin(), length, randchar );
        return str;
    }
}
