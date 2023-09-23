// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <regex>
#include <SoftVisionLog.h>

inline std::regex simpleFilterToRegex(const std::string& simpleFilter)
{
    std::string filterToRegex = simpleFilter;
    filterToRegex = std::regex_replace(filterToRegex, std::regex("/"), std::string("\\/"));
    filterToRegex = std::regex_replace(filterToRegex, std::regex("\\*"), std::string(".*"));
    filterToRegex = std::regex_replace(filterToRegex, std::regex("\\?"), std::string("."));
    filterToRegex = std::regex_replace(filterToRegex, std::regex("\\@"), std::string("[0-9]+")); // one @ correspond to one or more digits
    filterToRegex = std::regex_replace(filterToRegex, std::regex("\\#"), std::string("[0-9]"));  // each # in pattern correspond to a digit

    LOG_X("filterToRegex: " << filterToRegex);
    return std::regex(filterToRegex);
}

inline std::regex simpleFilterToRegex_noThrow(const std::string& simpleFilter)
{
    try
    {
        return simpleFilterToRegex(simpleFilter);
    }
    catch (std::regex_error& e)
    {
        LOG_X("Failed to create simpleFilterRegex from: " << simpleFilter << " => " << e.what());
        return std::regex();
    }
}


