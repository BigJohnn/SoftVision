// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sfmData/SfMData.hpp>
#include <feature/Regions.hpp>
#include <feature/RegionsPerView.hpp>
#include <feature/feature.hpp>
#include <matching/IndMatch.hpp>


namespace matching {


/**
 * @brief Compute the n best matches ('best' = mean of features' scale)
 * @param[in] inputMatches Set of indices for (putative) matches.
 * @param[in] regionsI Reference to the regions of the left image.
 * @param[in] regionsJ Reference to the regions of the right image.
 * @param[out] outputMatches Subset of inputMatches containing the best n matches, sorted.
 */
void sortMatches_byFeaturesScale(const matching::IndMatches& inputMatches,
                                 const feature::Regions& regionsI,
                                 const feature::Regions& regionsJ,
                                 matching::IndMatches& outputMatches);

/**
 * @brief Sort matches according to their Lowe ratio (ascending order).
 * @param[in,out] matches Set of indices for (putative) matches.
 */
void sortMatches_byDistanceRatio(matching::IndMatches& matches);

/**
 * @brief Compare method used in the match sorting.
 * @param[in] firstElem The first element to be compared.
 * @param[in] secondElem The second element to be compared.
 * @return True if firstElem is less than secondElem.
 */
bool matchCompare(const std::pair<float, size_t>& firstElem, const std::pair<float, size_t>& secondElem);

/**
 * @brief Extracts by copy the first (and best) uNumMatchesToKeep.
 * @param[out] outputMatches Set of image pairs and their respective sets of matches thresholded to the first
 * uNumMatchesToKeep.
 * @param[in] uNumMatchesToKeep The N best matches to keep.
 */
void thresholdMatches(matching::IndMatches& outputMatches, const std::size_t uNumMatchesToKeep);

/**
 * @brief Perform the gris filtering on the matches
 * @param[in] lRegions The regions of the first picture
 * @param[in] lImgSize Image size
 * @param[in] rRegions The regions of the second picture
 * @param[in] rImgSize Image size
 * @param[in] indexImagePair The Pair of matched images
 * @param[out] outMatches The remaining matches
 * @param[in] gridSize Number of cell per axis
 */
void matchesGridFiltering(const feature::Regions& lRegions,
                          const std::pair<std::size_t, std::size_t>& lImgSize,
                          const feature::Regions& rRegions,
                          const std::pair<std::size_t, std::size_t>& rImgSize,
                          const Pair& indexImagePair,
                          matching::IndMatches& outMatches, size_t gridSize = 3);

void matchesGridFilteringForAllPairs(const PairwiseMatches& geometricMatches,
                                     const sfmData::SfMData& sfmData,
                                     const feature::RegionsPerView& regionPerView,
                                     bool useGridSort, std::size_t numMatchesToKeep,
                                     PairwiseMatches& outPairwiseMatches);


void filterMatchesByMin2DMotion(PairwiseMatches& mapPutativesMatches,
                                const feature::RegionsPerView& regionPerView,
                                double minRequired2DMotion);

} // namespace sfm

