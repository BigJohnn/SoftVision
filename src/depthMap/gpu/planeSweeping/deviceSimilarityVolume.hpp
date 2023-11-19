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
#include <depthMap/gpu/planeSweeping/similarity.hpp>


namespace depthMap {

/**
 * @brief Initialize all the given similarity volume in device memory to the given value.
 * @param[in,out] inout_volume_dmp the similarity volume in device memory
 * @param[in] value the value to initalize with
 * @param[in] stream the stream for gpu execution
 */
extern void volumeInitialize(DeviceBuffer* inout_volume_dmp, TSim value);

/**
 * @brief Initialize all the given similarity volume in device memory to the given value.
 * @param[in,out] inout_volume_dmp the similarity volume in device memory
 * @param[in] value the value to initalize with
 * @param[in] stream the stream for gpu execution
 */
extern void volumeInitialize(DeviceBuffer* inout_volume_dmp, TSimRefine value);

/**
 * @brief Add similarity values from a given volume to another given volume.
 * @param[in,out] inout_volume_dmp the input/output similarity volume in device memory
 * @param[in] in_volume_dmp the input similarity volume in device memory
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_volumeAdd(DeviceBuffer* inout_volume_dmp, DeviceBuffer* in_volume_dmp);

/**
 * @brief Update second best similarity volume uninitialized values with first best volume values.
 * @param[in] in_volBestSim_dmp the best similarity volume in device memory
 * @param[out] inout_volSecBestSim_dmp the second best similarity volume in device memory
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_volumeUpdateUninitializedSimilarity(DeviceBuffer* in_volBestSim_dmp, DeviceBuffer* inout_volSecBestSim_dmp);

/**
 * @brief Compute the best / second best similarity volume for the given RC / TC.
 * @param[out] out_volBestSim_dmp the best similarity volume in device memory
 * @param[out] out_volSecBestSim_dmp the second best similarity volume in device memory
 * @param[in] in_depths_dmp the R camera depth list in device memory
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] tcDeviceCameraParamsId the T camera parameters id for array in device constant memory
 * @param[in] rcDeviceMipmapImage the R mipmap image in device memory container
 * @param[in] tcDeviceMipmapImage the T mipmap image in device memory container
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] depthRange the volume depth range to compute
 * @param[in] roi the 2d region of interest
 */
extern void volumeComputeSimilarity(DeviceBuffer* out_volBestSim_dmp,
                                         DeviceBuffer* out_volSecBestSim_dmp,
                                         DeviceBuffer* in_depths_dmp,
                                    id<MTLBuffer> rcDeviceCameraParams,
                                    id<MTLBuffer> tcDeviceCameraParams,
                                         const DeviceMipmapImage& rcDeviceMipmapImage,
                                         const DeviceMipmapImage& tcDeviceMipmapImage,
                                         const SgmParams& sgmParams, 
                                         const Range& depthRange,
                                         const ROI& roi);

/**
 * @brief Refine the best similarity volume for the given RC / TC.
 * @param[out] inout_volSim_dmp the similarity volume in device memory
 * @param[in] in_sgmDepthPixSizeMap_dmp the SGM upscaled depth/pixSize map (useful to get middle depth) in device memory
 * @param[in] in_sgmNormalMap_dmpPtr (or nullptr) the SGM upscaled normal map in device memory
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] tcDeviceCameraParamsId the T camera parameters id for array in device constant memory
 * @param[in] rcDeviceMipmapImage the R mipmap image in device memory container
 * @param[in] tcDeviceMipmapImage the T mipmap image in device memory container
 * @param[in] refineParams the Refine parameters
 * @param[in] depthRange the volume depth range to compute
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void volumeRefineSimilarity(DeviceBuffer* inout_volSim_dmp, 
                                        DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                        DeviceBuffer* in_sgmNormalMap_dmpPtr,
                                        const int rcDeviceCameraParamsId,
                                        const int tcDeviceCameraParamsId,
                                        const DeviceMipmapImage& rcDeviceMipmapImage,
                                        const DeviceMipmapImage& tcDeviceMipmapImage,
                                        const RefineParams& refineParams, 
                                        const Range& depthRange,
                                        const ROI& roi);

/**
 * @brief Filter / Optimize the given similarity volume
 * @param[out] out_volSimFiltered_dmp the output similarity volume in device memory
 * @param[in,out] inout_volSliceAccA_dmp the volume slice first accumulation buffer in device memory
 * @param[in,out] inout_volSliceAccB_dmp the volume slice second accumulation buffer in device memory
 * @param[in,out] inout_volAxisAcc_dmp the volume axisaccumulation buffer in device memory
 * @param[in] in_volSim_dmp the input similarity volume in device memory
 * @param[in] rcDeviceMipmapImage the R mipmap image in device memory container
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] lastDepthIndex the R camera last depth index
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_volumeOptimize(DeviceBuffer* out_volSimFiltered_dmp,
                                DeviceBuffer* inout_volSliceAccA_dmp,
                                DeviceBuffer* inout_volSliceAccB_dmp,
                                DeviceBuffer* inout_volAxisAcc_dmp,
                                DeviceBuffer* in_volSim_dmp, 
                                const DeviceMipmapImage& rcDeviceMipmapImage,
                                const SgmParams& sgmParams, 
                                const int lastDepthIndex,
                                const ROI& roi);

/**
 * @brief Retrieve the best depth/sim in the given similarity volume.
 * @param[out] out_sgmDepthThicknessMap_dmp the output depth/thickness map in device memory
 * @param[out] out_sgmDepthSimMap_dmp the output best depth/sim map in device memory
 * @param[in] in_depths_dmp the R camera depth list in device memory
 * @param[in] in_volSim_dmp the input similarity volume in device memory
 * @param[in] rcDeviceCameraParamsId the R camera parameters id for array in device constant memory
 * @param[in] sgmParams the Semi Global Matching parameters
 * @param[in] depthRange the volume depth range to compute
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_volumeRetrieveBestDepth(DeviceBuffer* out_sgmDepthThicknessMap_dmp,
                                         DeviceBuffer* out_sgmDepthSimMap_dmp,
                                         DeviceBuffer* in_depths_dmp,
                                         DeviceBuffer* in_volSim_dmp, 
                                         const int rcDeviceCameraParamsId,
                                         const SgmParams& sgmParams, 
                                         const Range& depthRange,
                                         const ROI& roi);

/**
 * @brief Retrieve the best depth/sim in the given refined similarity volume.
 * @param[out] out_refineDepthSimMap_dmp the output refined and fused depth/sim map in device memory
 * @param[in] in_sgmDepthPixSizeMap_dmp the SGM upscaled depth/pixSize map (useful to get middle depth) in device memory
 * @param[in] in_volSim_dmp the similarity volume in device memory
 * @param[in] refineParams the Refine parameters
 * @param[in] depthRange the volume depth range to compute
 * @param[in] roi the 2d region of interest
 * @param[in] stream the stream for gpu execution
 */
extern void cuda_volumeRefineBestDepth(DeviceBuffer* out_refineDepthSimMap_dmp,
                                       DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                       DeviceBuffer* in_volSim_dmp, 
                                       const RefineParams& refineParams, 
                                       const ROI& roi);

} // namespace depthMap

