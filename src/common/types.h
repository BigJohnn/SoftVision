//
//  types.h
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef types_h
#define types_h

#include <limits>
#include <set>

#include <cmath>
#define PI acos(-1)

#include <numeric/numeric.hpp>
typedef uint32_t IndexT;
typedef std::pair<IndexT,IndexT> Pair;
typedef std::set<Pair> PairSet;

static const IndexT UndefinedIndexT = std::numeric_limits<IndexT>::max();


//=============for Eigen Containers Storage=================
#include <map>
#include <Eigen/Core>
template<typename K, typename V>
using HashMap = std::map<K, V, std::less<K>, Eigen::aligned_allocator<std::pair<const K,V> > >;
//==========================================================



typedef std::pair<IndexT,IndexT> Pair;

struct EstimationStatus
{
  EstimationStatus(bool valid, bool strongSupport)
    : isValid(valid)
    , hasStrongSupport(strongSupport)
  {}

  bool isValid = false;
  bool hasStrongSupport = false;
};

#endif /* types_h */
