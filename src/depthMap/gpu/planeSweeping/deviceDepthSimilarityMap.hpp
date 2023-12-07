// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#import <depthMap/gpu/host/memory.hpp>

#include <mvsData/ROI.hpp>
#include <depthMap/SgmParams.hpp>
#include <depthMap/RefineParams.hpp>

#include <depthMap/gpu/host/DeviceMipmapImage.hpp>
#include <depthMap/gpu/device/DeviceCameraParams.hpp>


namespace depthMap {

/**
 * @brief Copy depth and default from input depth/sim map to another depth/sim map.
 * @param[out] out_depthSimMap_dmp the output depth/sim map
 * @param[in] in_depthSimMap_dmp the input depth/sim map to copy
 * @param[in] defaultSim the default similarity value to copy
 * @param[in] stream the stream for gpu execution
 */
extern void depthSimMapCopyDepthOnly(DeviceBuffer* out_depthSimMap_dmp,
                                          DeviceBuffer* in_depthSimMap_dmp,
                                          float defaultSim);

/**
 * @brief Upscale the given normal map.
 * @param[out] out_upscaledMap_dmp the output upscaled normal map
 * @param[in] in_map_dmp the normal map to upscaled
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void normalMapUpscale(DeviceBuffer* out_upscaledMap_dmp,
                                  DeviceBuffer* in_map_dmp,
                                  const ROI& roi);

/**
 * @brief Smooth thickness map with adjacent pixels.
 * @param[in,out] inout_depthThicknessMap_dmp the depth/thickness map
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] refineParams the Refine parameters
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void depthThicknessSmoothThickness(DeviceBuffer* inout_depthThicknessMap_dmp,
                                             const SgmParams& sgmParams,
                                             const RefineParams& refineParams,
                                             const ROI& roi);

/**
 * @brief Upscale the given depth/thickness map, filter masked pixels and compute pixSize from thickness.
 * @param[out] out_upscaledDepthPixSizeMap_dmp the output upscaled depth/pixSize map
 * @param[in] in_sgmDepthThicknessMap_dmp the input SGM depth/thickness map
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] rcDeviceMipmapImage the R mipmap image in device memory container
 * @param[in] refineParams the Refine parameters
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void computeSgmUpscaledDepthPixSizeMap(DeviceBuffer* out_upscaledDepthPixSizeMap_dmp,
                                                   DeviceBuffer* in_sgmDepthThicknessMap_dmp,
                                                DeviceCameraParams const& rcDeviceCameraParams,
                                                   const DeviceMipmapImage& rcDeviceMipmapImage,
                                                   const RefineParams& refineParams,
                                                   const ROI& roi);

/**
 * @brief Compute the normal map from the depth/sim map (only depth is used).
 * @param[out] out_normalMap_dmp the output normal map
 * @param[in] in_depthSimMap_dmp the input depth/sim map (only depth is used)
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] stepXY the input depth/sim map stepXY factor
 * @param[in] roi the 2d region of interest
 */
extern void depthSimMapComputeNormal(DeviceBuffer* out_normalMap_dmp,
                                          DeviceBuffer* in_depthSimMap_dmp,
                                          DeviceCameraParams const& rcDeviceCameraParams,
                                          const int stepXY,
                                          const ROI& roi);

/**
 * @brief Optimize a depth/sim map with the refineFused depth/sim map and the SGM depth/pixSize map.
 * @param[out] out_optimizeDepthSimMap_dmp the output optimized depth/sim map
 * @param[in,out] inout_imgVariance_dmp the image variance buffer
 * @param[in,out] inout_tmpOptDepthMap_dmp the temporary optimized depth map buffer
 * @param[in] in_sgmDepthPixSizeMap_dmp the input SGM upscaled depth/pixSize map
 * @param[in] in_refineDepthSimMap_dmp the input refined and fused depth/sim map
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] rcDeviceMipmapImage the R mipmap image in device memory container
 * @param[in] refineParams the Refine parameters
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_depthSimMapOptimizeGradientDescent(DeviceBuffer* out_optimizeDepthSimMap_dmp,
                                                    DeviceBuffer* inout_imgVariance_dmp,
                                                    DeviceBuffer* inout_tmpOptDepthMap_dmp,
                                                    DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                                    DeviceBuffer* in_refineDepthSimMap_dmp,
                                                    const int rcDeviceCameraParamsId,
                                                    const DeviceMipmapImage& rcDeviceMipmapImage,
                                                    const RefineParams& refineParams,
                                                    const ROI& roi);

} // namespace depthMap

