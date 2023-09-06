// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "matching/matcherType.hpp"
#include "matching/RegionsMatcher.hpp"
#include "matching/ArrayMatcher_bruteForce.hpp"
#include "matching/ArrayMatcher_kdtreeFlann.hpp"
#include "matching/ArrayMatcher_cascadeHashing.hpp"

#include <SoftVisionLog.h>


namespace matching {

void DistanceRatioMatch(
  std::mt19937 & randomNumberGenerator,
  float f_dist_ratio,
  matching::EMatcherType eMatcherType,
  const feature::Regions & regions_I, // database
  const feature::Regions & regions_J, // query
  matching::IndMatches & matches)
{
  RegionsDatabaseMatcher matcher(randomNumberGenerator, eMatcherType, regions_I);
  matcher.Match(f_dist_ratio, regions_J, matches);
}

bool RegionsDatabaseMatcher::Match(
  float distRatio,
  const feature::Regions & queryRegions,
  matching::IndMatches & matches) const
{
  if (queryRegions.RegionCount() == 0)
    return false;

  if (!_regionsMatcher)
    return false;

  return _regionsMatcher->Match(distRatio, queryRegions, matches);
}

RegionsDatabaseMatcher::RegionsDatabaseMatcher():
  _matcherType(BRUTE_FORCE_L2),
  _regionsMatcher(nullptr)
{}


RegionsDatabaseMatcher::RegionsDatabaseMatcher(
  std::mt19937 & randomNumberGenerator,
  matching::EMatcherType matcherType,
  const feature::Regions & databaseRegions)
  : _matcherType(matcherType)
{
  _regionsMatcher = createRegionsMatcher(randomNumberGenerator, databaseRegions, matcherType);
}


std::unique_ptr<IRegionsMatcher> createRegionsMatcher(std::mt19937 & randomNumberGenerator,const feature::Regions & regions, matching::EMatcherType matcherType)
{
  std::unique_ptr<IRegionsMatcher> out;

  // Handle invalid request
  if (regions.IsScalar() && matcherType == BRUTE_FORCE_HAMMING)
    return out;
  if (regions.IsBinary() && matcherType != BRUTE_FORCE_HAMMING)
    return out;

  // Switch regions type ID, matcher & Metric: initialize the Matcher interface
  if (regions.IsScalar())
  {
    if (regions.Type_id() == typeid(unsigned char).name())
    {
      // Build on the fly unsigned char based Matcher
      switch (matcherType)
      {
        case BRUTE_FORCE_L2:
        {
          typedef feature::L2_Vectorized<unsigned char> MetricT;
          typedef ArrayMatcher_bruteForce<unsigned char, MetricT> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case ANN_L2:
        {
          typedef ArrayMatcher_kdtreeFlann<unsigned char> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case CASCADE_HASHING_L2:
        {
          typedef feature::L2_Vectorized<unsigned char> MetricT;
          typedef ArrayMatcher_cascadeHashing<unsigned char, MetricT> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        default:
          LOG_INFO("Using unknown matcher type");
      }
    }
    else if (regions.Type_id() == typeid(float).name())
    {
      // Build on the fly float based Matcher
      switch (matcherType)
      {
        case BRUTE_FORCE_L2:
        {
          typedef feature::L2_Vectorized<float> MetricT;
          typedef ArrayMatcher_bruteForce<float, MetricT> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case ANN_L2:
        {
          typedef ArrayMatcher_kdtreeFlann<float> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case CASCADE_HASHING_L2:
        {
          typedef feature::L2_Vectorized<float> MetricT;
          typedef ArrayMatcher_cascadeHashing<float, MetricT> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        default:
          LOG_INFO("Using unknown matcher type");
      }
    }
    else if (regions.Type_id() == typeid(double).name())
    {
      // Build on the fly double based Matcher
      switch (matcherType)
      {
        case BRUTE_FORCE_L2:
        {
          typedef feature::L2_Vectorized<double> MetricT;
          typedef ArrayMatcher_bruteForce<double, MetricT> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case ANN_L2:
        {
          typedef ArrayMatcher_kdtreeFlann<double> MatcherT;
          out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
        }
        break;
        case CASCADE_HASHING_L2:
        {
          LOG_INFO("Not yet implemented");
        }
        break;
        default:
          LOG_INFO("Using unknown matcher type");
      }
    }
  }
  else if (regions.IsBinary() && regions.Type_id() == typeid(unsigned char).name())
  {
    switch (matcherType)
    {
      case BRUTE_FORCE_HAMMING:
      {
        typedef feature::Hamming<unsigned char> Metric;
        typedef ArrayMatcher_bruteForce<unsigned char, Metric> MatcherT;
        out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, false));
      }
      break;
      default:
          LOG_INFO("Using unknown matcher type");
    }
  }
  else
  {
    LOG_INFO("Please consider add this region type_id to Matcher_Regions_Database::Match(...)\n typeid: %s", regions.Type_id().c_str());
  }
  return out;
}

}  // namespace matching

