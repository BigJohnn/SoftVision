#pragma once

#include <unistd.h>
#include <string>
#include <sys/stat.h>
#include <SoftVisionLog.h>
namespace utils {
    bool exists(std::string const& pathname)
    {
        return 0 == access(pathname.c_str(), F_OK);
    }

    bool create_directory(std::string const& pathname)
    {
        if(exists(pathname)) {
            LOG_INFO("%s already exist!", pathname.c_str());
            return true;
        }
        return 0 == mkdir(pathname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    }
}
