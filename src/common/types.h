//
//  types.h
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef types_h
#define types_h

#include <limits>

typedef uint32_t IndexT;
static const IndexT UndefinedIndexT = std::numeric_limits<IndexT>::max();

//=============for Eigen Containers Storage=================
#include <map>
#include <Eigen/Core>
template<typename K, typename V>
using HashMap = std::map<K, V, std::less<K>, Eigen::aligned_allocator<std::pair<const K,V> > >;
//==========================================================

#include <cmath>
#define PI acos(-1)

typedef std::pair<IndexT,IndexT> Pair;

#endif /* types_h */
