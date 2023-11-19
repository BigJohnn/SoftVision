// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#import <depthMap/gpu/host/memory.hpp>

#include <mvsData/ROI.hpp>
#include <mvsUtils/MultiViewParams.hpp>
#include <depthMap/SgmParams.hpp>
#include <depthMap/RefineParams.hpp>

#include <depthMap/gpu/planeSweeping/similarity.hpp>

#include <string>
#include <vector>


namespace depthMap {

/**
 * @brief Export 9 similarity values over the entire depth in a CSV file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depths the SGM depth list
 * @param[in] name the export name
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilaritySamplesCSV(DeviceBuffer* in_volumeSim_hmh,
                                const std::vector<float>& in_depths,
                                const std::string& name, 
                                const SgmParams& sgmParams,
                                const std::string& filepath,
                                const ROI& roi);

/**
 * @brief Export 9 similarity values over the entire depth in a CSV file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] name the export name
 * @param[in] refineParams the Refine parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilaritySamplesCSV(DeviceBuffer* in_volumeSim_hmh,
                                const std::string& name, 
                                const RefineParams& refineParams,
                                const std::string& filepath,
                                const ROI& roi);

/**
 * @brief Export the given similarity volume to an Alembic file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depths the SGM depth list
 * @param[in] mp the multi-view parameters
 * @param[in] camIndex the R cam global index
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilarityVolume(DeviceBuffer* in_volumeSim_hmh,
                            const std::vector<float>& in_depths,
                            const mvsUtils::MultiViewParams& mp, 
                            int camIndex, 
                            const SgmParams& sgmParams,
                            const std::string& filepath,
                            const ROI& roi);

/**
 * @brief Export a cross of the given similarity volume to an Alembic file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depths the SGM depth list
 * @param[in] mp the multi-view parameters
 * @param[in] camIndex the R cam global index
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilarityVolumeCross(DeviceBuffer* in_volumeSim_hmh,
                                 const std::vector<float>& in_depths,
                                 const mvsUtils::MultiViewParams& mp, 
                                 int camIndex, 
                                 const SgmParams& sgmParams,
                                 const std::string& filepath, 
                                 const ROI& roi);

/**
 * @brief Export a cross of the given similarity volume to an Alembic file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depthSimMapSgmUpscale_hmh the upscaled SGM depth/sim map
 * @param[in] mp the multi-view parameters
 * @param[in] camIndex the R cam global index
 * @param[in] refineParams the Refine parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilarityVolumeCross(DeviceBuffer* in_volumeSim_hmh,
                                 DeviceBuffer* in_depthSimMapSgmUpscale_hmh,
                                 const mvsUtils::MultiViewParams& mp, 
                                 int camIndex,
                                 const RefineParams& refineParams, 
                                 const std::string& filepath, 
                                 const ROI& roi);

/**
 * @brief Export a topographic cut of the given similarity volume to an Alembic file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depths the SGM depth list
 * @param[in] mp the multi-view parameters
 * @param[in] camIndex the R cam global index
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilarityVolumeTopographicCut(DeviceBuffer* in_volumeSim_hmh,
                                          const std::vector<float>& in_depths,
                                          const mvsUtils::MultiViewParams& mp,
                                          int camIndex,
                                          const SgmParams& sgmParams,
                                          const std::string& filepath,
                                          const ROI& roi);

/**
 * @brief Export a topographic cut of the given similarity volume to an Alembic file.
 * @param[in] in_volumeSim_hmh the similarity in host memory
 * @param[in] in_depthSimMapSgmUpscale_hmh the upscaled SGM depth/sim map
 * @param[in] mp the multi-view parameters
 * @param[in] camIndex the R cam global index
 * @param[in] refineParams the Refine parameters
 * @param[in] filepath the export filepath
 * @param[in] roi the 2d region of interest
 */
void exportSimilarityVolumeTopographicCut(DeviceBuffer* in_volumeSim_hmh,
                                          DeviceBuffer* in_depthSimMapSgmUpscale_hmh,
                                          const mvsUtils::MultiViewParams& mp,
                                          int camIndex,
                                          const RefineParams& refineParams,
                                          const std::string& filepath,
                                          const ROI& roi);

/**
 * @brief Export the given similarity volume to an Alembic file.
 */
void exportColorVolume(DeviceBuffer* in_volumeSim_hmh, 
                       const std::vector<float>& in_depths,
                       int startDepth, 
                       int nbDepths, 
                       const mvsUtils::MultiViewParams& mp, 
                       int camIndex, 
                       int scale, 
                       int step, 
                       const std::string& filepath, 
                       const ROI& roi);

} // namespace depthMap

