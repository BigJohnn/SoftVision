#pragma once

#include <unistd.h>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
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

    bool is_directory(std::string const& pathname)
    {
        struct stat s;
        bool ret = false;
        if( stat(pathname.c_str(),&s) == 0 )
        {
            if( s.st_mode & S_IFDIR )
            {
                // it's a directory
                ret = true;
            }
//            else if( s.st_mode & S_IFREG )
//            {
//                // it's a file
//            }
        }
        
        return ret;
    }

    bool is_empty(std::string const& dirname)
    {
        DIR *dp;
        struct dirent *entry;

        dp = opendir(dirname.c_str());
        if (dp == NULL) {
            perror("opendir");
            return 1;
        }

        while ((entry = readdir(dp)) != NULL) {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                printf("Directory is not empty\n");
                closedir(dp);
                return false;
            }
        }

        printf("Directory is empty\n");
        closedir(dp);
        
        return true;
    }

    bool remove_all(const std::string& depthMapsPtsSimsTmpDir) {
        LOG_ERROR("remove_all TODO: impl");
        return true;
    }

    bool remove(const std::string& filePath) {
        LOG_ERROR("remove TODO: impl");
        return true;
    }
}
