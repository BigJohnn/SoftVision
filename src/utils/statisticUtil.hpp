#pragma once
#include <algorithm>
#include <vector>
#include <cmath>

namespace utils {

//TODO: check
//    percentile*100 % of distribution is below retVal:
template <typename T>
double quantile(std::vector<T>& distribute, double percentile, int cache_size = INT_MAX, bool right_tail = true)
{
    std::sort(distribute.begin(), distribute.end());
    
    std::vector<double> dist;
    
    if(cache_size < distribute.size()) {
        if(!right_tail) {
            dist = std::vector<double>(distribute.begin(), distribute.begin() + cache_size);
        }
        else {
            dist = std::vector<double>(distribute.end() - cache_size, distribute.end());
        }
    }
    else {
        dist = distribute;
    }
    
    return dist[floor(percentile * (dist.size()-1))];
}

}
