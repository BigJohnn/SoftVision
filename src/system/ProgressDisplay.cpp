// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ProgressDisplay.hpp"
//#include <boost/timer/progress_display.hpp>
#include <mutex>

namespace system2 {

ProgressDisplayImpl::~ProgressDisplayImpl() = default;

class ProgressDisplayImplEmpty : public ProgressDisplayImpl {
public:
    void restart(unsigned long expectedCount) override {}
    void increment(unsigned long count) override {}
    unsigned long count() override { return 0; }
    unsigned long expectedCount() override { return 0; }
};

ProgressDisplay::ProgressDisplay() : _impl{std::make_shared<ProgressDisplayImplEmpty>()}
{}

class ProgressDisplayImplBoostProgress : public ProgressDisplayImpl {
public:
    ProgressDisplayImplBoostProgress(unsigned long expectedCount,
                                     std::ostream& os,
                                     const std::string& s1,
                                     const std::string& s2,
                                     const std::string& s3) :
        m_expectedCount(expectedCount)
    {
    }

    ~ProgressDisplayImplBoostProgress() override = default;

    void restart(unsigned long expectedCount) override
    {
        m_count = 0;
        m_expectedCount = expectedCount;
//        _display.restart(expectedCount);
    }

    void increment(unsigned long count) override
    {
        std::lock_guard<std::mutex> lock{_mutex};
        m_count += count;
//        _display += count;
    }

    unsigned long count() override
    {
        std::lock_guard<std::mutex> lock{_mutex};
        return m_count;
//        return _display.count();
    }

    unsigned long expectedCount() override
    {
        return m_expectedCount;
    }

private:
    std::mutex _mutex;
    unsigned long m_expectedCount;
    unsigned long m_count;
//    boost::timer::progress_display _display;
};


ProgressDisplay createConsoleProgressDisplay(unsigned long expectedCount,
                                             std::ostream& os,
                                             const std::string& s1,
                                             const std::string& s2,
                                             const std::string& s3)
{
    auto impl = std::make_shared<ProgressDisplayImplBoostProgress>(expectedCount, os, s1, s2, s3);
    return ProgressDisplay(impl);
}

} // namespace system2
