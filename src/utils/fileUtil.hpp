#pragma once

#include <unistd.h>
#include <string>
#include <sys/stat.h>

namespace utils {
    bool exists(std::string const& pathname)
    {
        return 0 == access(pathname.c_str(), F_OK);
    }

    bool create_directory(std::string const& pathname)
    {
        return 0 == mkdir(pathname.c_str(), S_IFDIR);
    }
}
