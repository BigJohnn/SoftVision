#include <algorithm>
#include <vector>
#incldue <cmath>

namespace utils {

//TODO: check
//    percentile*100 % of distribution is below retVal:
double quantile(std::vector<double>& distribute, double percentile)
{
    std::sort(distribute.begin(), distribute.end());
    return distribute[floor(percentile * (distribute.size()-1))];
}

}
