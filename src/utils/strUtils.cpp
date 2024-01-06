#include <utils/strUtils.hpp>

namespace utils{

static std::string s_temp_path;

std::string& temp_directory_path()
{
    return s_temp_path;
}

}
