// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <matchingImageCollection/ImageCollectionMatcher_generic.hpp>
#include <matching/ArrayMatcher_bruteForce.hpp>
#include <matching/ArrayMatcher_kdtreeFlann.hpp>
#include <matching/ArrayMatcher_cascadeHashing.hpp>
#include <matching/RegionsMatcher.hpp>
#include <matchingImageCollection/IImageCollectionMatcher.hpp>
#include <system/ProgressDisplay.hpp>
//#include <config.hpp>

#include <SoftVisionLog.h>


namespace matchingImageCollection {

using namespace matching;
using namespace feature;

ImageCollectionMatcher_generic::ImageCollectionMatcher_generic(
  float distRatio, bool crossMatching, EMatcherType matcherType)
  : IImageCollectionMatcher()
  , _f_dist_ratio(distRatio)
  , _useCrossMatching(crossMatching)
  , _matcherType(matcherType)
{
}

void ImageCollectionMatcher_generic::Match(
  std::mt19937 & randomNumberGenerator,
  const feature::RegionsPerView& regionsPerView,
  const PairSet & pairs,
  feature::EImageDescriberType descType,
  matching::PairwiseMatches & map_PutativesMatches)const // the pairwise photometric corresponding points
{

  LOG_DEBUG("Using the OPENMP thread interface");

  const bool b_multithreaded_pair_search = (_matcherType == CASCADE_HASHING_L2);
  // -> set to true for CASCADE_HASHING_L2, since OpenMP instructions are not used in this matcher

  auto progressDisplay = system2::createConsoleProgressDisplay(pairs.size(), std::cout);

  // Sort pairs according the first index to minimize the MatcherT build operations
  typedef std::map<size_t, std::vector<size_t> > Map_vectorT;
  Map_vectorT map_Pairs;
  for (PairSet::const_iterator iter = pairs.begin(); iter != pairs.end(); ++iter)
  {
    map_Pairs[iter->first].push_back(iter->second);
  }

  // Perform matching between all the pairs
  for (Map_vectorT::const_iterator iter = map_Pairs.begin();
    iter != map_Pairs.end(); ++iter)
  {
    const size_t I = iter->first;
    const std::vector<size_t> & indexToCompare = iter->second;

    const feature::Regions & regionsI = regionsPerView.getRegions(I, descType);

    if (regionsI.RegionCount() == 0)
    {
      progressDisplay += indexToCompare.size();
      continue;
    }

    // Initialize the matching interface
    matching::RegionsDatabaseMatcher matcher(randomNumberGenerator, _matcherType, regionsI);

    #pragma omp parallel for schedule(dynamic) if(b_multithreaded_pair_search)
    for (int j = 0; j < (int)indexToCompare.size(); ++j)
    {
      const size_t J = indexToCompare[j];

        LOG_DEBUG("thread %d is running number %d", omp_get_thread_num(), j);
      const feature::Regions &regionsJ = regionsPerView.getRegions(J, descType);
      if (regionsJ.RegionCount() == 0
          || regionsI.Type_id() != regionsJ.Type_id())
      {
        ++progressDisplay;
        continue;
      }

      IndMatches vec_putatives_matches;
      matcher.Match(_f_dist_ratio, regionsJ, vec_putatives_matches);

      if (_useCrossMatching)
      {
        // Initialize the matching interface
        matching::RegionsDatabaseMatcher matcherCross(randomNumberGenerator, _matcherType, regionsJ);  

        IndMatches vec_putatives_matches_cross;
        matcherCross.Match(_f_dist_ratio, regionsI, vec_putatives_matches_cross);

        //Create a dictionnary of matches indexed by their pair of indexes
        std::map<std::pair<int, int>, IndMatch> check_matches;
        for (IndMatch & m : vec_putatives_matches_cross)
        {
          std::pair<int, int> key = std::make_pair(m._i, m._j);
          check_matches[key] = m;
        }

        IndMatches vec_putatives_matches_checked;
        for (IndMatch & m : vec_putatives_matches)
        {
          //Check with reversed key (images are swapped)
          std::pair<int, int> key = std::make_pair(m._j, m._i);
          if (check_matches.find(key) != check_matches.end())
          {
            vec_putatives_matches_checked.push_back(m);
          }
        }

        std::swap(vec_putatives_matches, vec_putatives_matches_checked);
      }

      #pragma omp critical
      {
        ++progressDisplay;
        if (!vec_putatives_matches.empty())
        {
          map_PutativesMatches[std::make_pair(I,J)].emplace(descType, std::move(vec_putatives_matches));
        }
      }
    }
  }
}


} // namespace matchingImageCollection
