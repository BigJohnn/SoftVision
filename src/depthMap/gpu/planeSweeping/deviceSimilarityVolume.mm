// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
#import <depthMap/gpu/host/ComputePipeline.hpp>

#include "deviceSimilarityVolume.hpp"
#include "deviceSimilarityVolumeKernels.cuh"

#include <mvsData/ROI_d.hpp>
//#include <depthMap/gpu/host/divUp.hpp>

#include <map>


namespace depthMap {

/**
 * @brief Get maximum potential block size for the given kernel function.
 *        Provides optimal block size based on the capacity of the device.
 *
 * @see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
 *
 * @param[in] kernelFuction the given kernel function
 *
 * @return recommended or default block size for kernel execution
 */
template<class T>
dim3 getMaxPotentialBlockSize(T kernelFuction)
{
    const dim3 defaultBlock(32, 1, 1); // minimal default settings

    int recommendedMinGridSize;
    int recommendedBlockSize;

    cudaError_t err;
    err = cudaOccupancyMaxPotentialBlockSize(&recommendedMinGridSize,
                                             &recommendedBlockSize,
                                             kernelFuction,
                                             0, // dynamic shared mem size: none used
                                             0); // no block size limit, 1 thread OK

    if(err != cudaSuccess)
    {
        ALICEVISION_LOG_WARNING( "cudaOccupancyMaxPotentialBlockSize failed, using default block settings.");
        return defaultBlock;
    }

    if(recommendedBlockSize > 32)
    {
        const dim3 recommendedBlock(32, divUp(recommendedBlockSize, 32), 1);
        return recommendedBlock;
    }

    return defaultBlock;
}

void cuda_volumeInitialize(DeviceBuffer* inout_volume_dmp, TSim value)
{
    // get input/output volume dimensions
    MTLSize volDim = [inout_volume_dmp getSize];

    // kernel launch parameters
    MTLSize block = MTLSizeMake(32, 4, 1);
    MTLSize grid = MTLSizeMake(volDim.width, volDim.height, volDim.depth);
//    const dim3 block(32, 4, 1);
//    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());

    // kernel execution
    
//    [inout_volume_dmp allocate:volDim elemSizeInBytes:sizeof(float)];
    NSArray* args = @[
        [inout_volume_dmp getBuffer],
        [NSNumber numberWithInt:[inout_volume_dmp getBytesUpToDim:1]], //1024*256
        [NSNumber numberWithInt:[inout_volume_dmp getBytesUpToDim:0]], // 1024
        [NSNumber numberWithUnsignedInt:(unsigned)volDim.width],
        [NSNumber numberWithUnsignedInt:(unsigned)volDim.height],
        @255.f
    ];
    
    [ComputePipeline Exec:grid ThreadgroupSize:block KernelFuncName:@"volume_init_kernel" Args:args];
    
//    volume_init_kernel<TSim><<<grid, block, 0, stream>>>(
//        inout_volume_dmp.getBuffer(),
//        inout_volume_dmp.getBytesPaddedUpToDim(1),
//        inout_volume_dmp.getBytesPaddedUpToDim(0),
//        (unsigned int)(volDim.x()),
//        (unsigned int)(volDim.y()),
//        value);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void cuda_volumeInitialize(DeviceBuffer* inout_volume_dmp, TSimRefine value)
{
    // get input/output volume dimensions
//    const CudaSize<3>& volDim = inout_volume_dmp.getSize();
//
//    // kernel launch parameters
//    const dim3 block(32, 4, 1);
//    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());
    
    // get input/output volume dimensions
    MTLSize volDim = [inout_volume_dmp getSize];

    // kernel launch parameters
    MTLSize block = MTLSizeMake(32, 4, 1);
    MTLSize grid = MTLSizeMake(volDim.width, volDim.height, volDim.depth);

    // kernel execution
    NSArray* args = @[
        [inout_volume_dmp getBuffer], //TODO: check: TSimRefine是否使用half
        [NSNumber numberWithInt:[inout_volume_dmp getBytesUpToDim:1]], //1024*256
        [NSNumber numberWithInt:[inout_volume_dmp getBytesUpToDim:0]], // 1024
        [NSNumber numberWithUnsignedInt:(unsigned)volDim.width],
        [NSNumber numberWithUnsignedInt:(unsigned)volDim.height],
        @255.f
    ];
    
    [ComputePipeline Exec:grid ThreadgroupSize:block KernelFuncName:@"volume_init_kernel" Args:args];
    
//    volume_init_kernel<TSimRefine><<<grid, block, 0, stream>>>(
//        inout_volume_dmp.getBuffer(),
//        inout_volume_dmp.getBytesPaddedUpToDim(1),
//        inout_volume_dmp.getBytesPaddedUpToDim(0),
//        (unsigned int)(volDim.x()),
//        (unsigned int)(volDim.y()),
//        value);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void cuda_volumeAdd(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp,
                             const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volume_dmp, 
                             cudaStream_t stream)
{
    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const dim3 block(32, 4, 1);
    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());

    // kernel execution
    volume_add_kernel<<<grid, block, 0, stream>>>(
        inout_volume_dmp.getBuffer(),
        inout_volume_dmp.getBytesPaddedUpToDim(1),
        inout_volume_dmp.getBytesPaddedUpToDim(0),
        in_volume_dmp.getBuffer(),
        in_volume_dmp.getBytesPaddedUpToDim(1),
        in_volume_dmp.getBytesPaddedUpToDim(0),
        (unsigned int)(volDim.x()),
        (unsigned int)(volDim.y()));

    // check cuda last error
    CHECK_CUDA_ERROR();
}

void cuda_volumeUpdateUninitializedSimilarity(const CudaDeviceMemoryPitched<TSim, 3>& in_volBestSim_dmp,
                                                       CudaDeviceMemoryPitched<TSim, 3>& inout_volSecBestSim_dmp,
                                                       cudaStream_t stream)
{
    assert(in_volBestSim_dmp.getSize() == inout_volSecBestSim_dmp.getSize());

    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volSecBestSim_dmp.getSize();

    // kernel launch parameters
    const dim3 block = getMaxPotentialBlockSize(volume_updateUninitialized_kernel);
    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());

    // kernel execution
    volume_updateUninitialized_kernel<<<grid, block, 0, stream>>>(
        inout_volSecBestSim_dmp.getBuffer(),
        inout_volSecBestSim_dmp.getBytesPaddedUpToDim(1),
        inout_volSecBestSim_dmp.getBytesPaddedUpToDim(0),
        in_volBestSim_dmp.getBuffer(),
        in_volBestSim_dmp.getBytesPaddedUpToDim(1),
        in_volBestSim_dmp.getBytesPaddedUpToDim(0), 
        (unsigned int)(volDim.x()),
        (unsigned int)(volDim.y()));

    // check cuda last error
    CHECK_CUDA_ERROR();
}

void volumeComputeSimilarity(DeviceBuffer* out_volBestSim_dmp,
                              DeviceBuffer* out_volSecBestSim_dmp,
                               DeviceBuffer* in_depths_dmp,
                               DeviceBuffer* rcDeviceCameraParams,
                               DeviceBuffer* tcDeviceCameraParams,
                               const DeviceMipmapImage& rcDeviceMipmapImage,
                               const DeviceMipmapImage& tcDeviceMipmapImage,
                               const SgmParams& sgmParams,
                               const Range& depthRange,
                               const ROI& roi)
{
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale);
    MTLSize rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);
    MTLSize tcLevelDim = tcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // kernel launch parameters
//    const dim3 block = getMaxPotentialBlockSize(volume_computeSimilarity_kernel);
//    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), depthRange.size());
    
    MTLSize block = MTLSizeMake(32, 4, 1); // TODO: check!
    MTLSize grid = MTLSizeMake(roi.width(), roi.height(), depthRange.size());

    // kernel execution
    
//    [inout_volume_dmp allocate:volDim elemSizeInBytes:sizeof(float)];
    Range_d depthRange_d(depthRange.begin, depthRange.end);
    ROI_d roi_d(roi.x.begin, roi.y.begin, roi.x.end, roi.y.end);
    NSArray* args = @[
        [out_volBestSim_dmp getBuffer],
        [NSNumber numberWithInt:[out_volBestSim_dmp getBytesUpToDim:1]], //1024*256
        [NSNumber numberWithInt:[out_volBestSim_dmp getBytesUpToDim:0]], // 1024
        [out_volSecBestSim_dmp getBuffer],
        [NSNumber numberWithInt:[out_volSecBestSim_dmp getBytesUpToDim:1]], //1024*256
        [NSNumber numberWithInt:[out_volSecBestSim_dmp getBytesUpToDim:0]], // 1024
        [in_depths_dmp getBuffer],
        [NSNumber numberWithInt:[in_depths_dmp getBytesUpToDim:0]], // 1024
        rcDeviceCameraParams,
        tcDeviceCameraParams,
        rcDeviceMipmapImage.getTextureObject(),
        tcDeviceMipmapImage.getTextureObject(),
        [NSNumber numberWithUnsignedInt:(unsigned)rcLevelDim.width],
        [NSNumber numberWithUnsignedInt:(unsigned)rcLevelDim.height],
        [NSNumber numberWithUnsignedInt:(unsigned)tcLevelDim.width],
        [NSNumber numberWithUnsignedInt:(unsigned)tcLevelDim.height],
        [NSNumber numberWithInt:rcMipmapLevel],
        [NSNumber numberWithInt:sgmParams.stepXY],
        [NSNumber numberWithInt:sgmParams.wsh],
        
        [NSNumber numberWithFloat:1.f / float(sgmParams.gammaC)],
        [NSNumber numberWithInt:1.f / float(sgmParams.gammaP)],
        
        [NSNumber numberWithBool:sgmParams.useConsistentScale],
        [NSNumber numberWithBool:sgmParams.useCustomPatchPattern],
        depthRange_d,
        roi_d
    ];
    
    [ComputePipeline Exec:grid ThreadgroupSize:block KernelFuncName:@"volume_computeSimilarity_kernel" Args:args];

    // kernel execution
//    volume_computeSimilarity_kernel<<<grid, block, 0, stream>>>(
//        out_volBestSim_dmp.getBuffer(),
//        out_volBestSim_dmp.getBytesPaddedUpToDim(1),
//        out_volBestSim_dmp.getBytesPaddedUpToDim(0),
//        out_volSecBestSim_dmp.getBuffer(),
//        out_volSecBestSim_dmp.getBytesPaddedUpToDim(1),
//        out_volSecBestSim_dmp.getBytesPaddedUpToDim(0),
//        in_depths_dmp.getBuffer(),
//        in_depths_dmp.getBytesPaddedUpToDim(0),
//        rcDeviceCameraParamsId,
//        tcDeviceCameraParamsId,
//        rcDeviceMipmapImage.getTextureObject(),
//        tcDeviceMipmapImage.getTextureObject(),
//        (unsigned int)(rcLevelDim.x()),
//        (unsigned int)(rcLevelDim.y()),
//        (unsigned int)(tcLevelDim.x()),
//        (unsigned int)(tcLevelDim.y()),
//        rcMipmapLevel,
//        sgmParams.stepXY,
//        sgmParams.wsh,
//        (1.f / float(sgmParams.gammaC)), // inverted gammaC
//        (1.f / float(sgmParams.gammaP)), // inverted gammaP
//        sgmParams.useConsistentScale,
//        sgmParams.useCustomPatchPattern,
//        depthRange,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

extern void cuda_volumeRefineSimilarity(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volSim_dmp, 
                                        const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                        const CudaDeviceMemoryPitched<float3, 2>* in_sgmNormalMap_dmpPtr,
                                        const int rcDeviceCameraParamsId,
                                        const int tcDeviceCameraParamsId,
                                        const DeviceMipmapImage& rcDeviceMipmapImage,
                                        const DeviceMipmapImage& tcDeviceMipmapImage,
                                        const RefineParams& refineParams, 
                                        const Range& depthRange,
                                        const ROI& roi,
                                        cudaStream_t stream)
{
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
    const CudaSize<2> tcLevelDim = tcDeviceMipmapImage.getDimensions(refineParams.scale);

    // kernel launch parameters
    const dim3 block = getMaxPotentialBlockSize(volume_refineSimilarity_kernel);
    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), depthRange.size());

    // kernel execution
    volume_refineSimilarity_kernel<<<grid, block, 0, stream>>>(
        inout_volSim_dmp.getBuffer(),
        inout_volSim_dmp.getBytesPaddedUpToDim(1),
        inout_volSim_dmp.getBytesPaddedUpToDim(0),
        in_sgmDepthPixSizeMap_dmp.getBuffer(),
        in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0),
        (in_sgmNormalMap_dmpPtr == nullptr) ? nullptr : in_sgmNormalMap_dmpPtr->getBuffer(),
        (in_sgmNormalMap_dmpPtr == nullptr) ? 0 : in_sgmNormalMap_dmpPtr->getBytesPaddedUpToDim(0),
        rcDeviceCameraParamsId,
        tcDeviceCameraParamsId,
        rcDeviceMipmapImage.getTextureObject(),
        tcDeviceMipmapImage.getTextureObject(),
        (unsigned int)(rcLevelDim.x()),
        (unsigned int)(rcLevelDim.y()),
        (unsigned int)(tcLevelDim.x()),
        (unsigned int)(tcLevelDim.y()),
        rcMipmapLevel,
        int(inout_volSim_dmp.getSize().z()), 
        refineParams.stepXY,
        refineParams.wsh, 
        (1.f / float(refineParams.gammaC)), // inverted gammaC
        (1.f / float(refineParams.gammaP)), // inverted gammaP
        refineParams.useConsistentScale,
        refineParams.useCustomPatchPattern,
        depthRange,
        roi);

    // check cuda last error
    CHECK_CUDA_ERROR();
}


void cuda_volumeAggregatePath(CudaDeviceMemoryPitched<TSim, 3>& out_volAgr_dmp,
                                       CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccA_dmp,
                                       CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccB_dmp,
                                       CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volAxisAcc_dmp,
                                       const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp, 
                                       const DeviceMipmapImage& rcDeviceMipmapImage,
                                       const CudaSize<2>& rcLevelDim,
                                       const float rcMipmapLevel,
                                       const CudaSize<3>& axisT,
                                       const SgmParams& sgmParams,
                                       const int lastDepthIndex,
                                       const int filteringIndex,
                                       const bool invY,
                                       const ROI& roi,
                                       cudaStream_t stream)
{
    CudaSize<3> volDim = in_volSim_dmp.getSize();
    volDim[2] = lastDepthIndex; // override volume depth, use rc depth list last index

    const size_t volDimX = volDim[axisT[0]];
    const size_t volDimY = volDim[axisT[1]];
    const size_t volDimZ = volDim[axisT[2]];

    const int3 volDim_ = make_int3(volDim[0], volDim[1], volDim[2]);
    const int3 axisT_ = make_int3(axisT[0], axisT[1], axisT[2]);
    const int ySign = (invY ? -1 : 1);

    // setup block and grid
    const int blockSize = 8;
    const dim3 blockVolXZ(blockSize, blockSize, 1);
    const dim3 gridVolXZ(divUp(volDimX, blockVolXZ.x), divUp(volDimZ, blockVolXZ.y), 1);

    const int blockSizeL = 64;
    const dim3 blockColZ(blockSizeL, 1, 1);
    const dim3 gridColZ(divUp(volDimX, blockColZ.x), 1, 1);

    const dim3 blockVolSlide(blockSizeL, 1, 1);
    const dim3 gridVolSlide(divUp(volDimX, blockVolSlide.x), volDimZ, 1);

    CudaDeviceMemoryPitched<TSimAcc, 2>* xzSliceForY_dmpPtr   = &inout_volSliceAccA_dmp; // Y slice
    CudaDeviceMemoryPitched<TSimAcc, 2>* xzSliceForYm1_dmpPtr = &inout_volSliceAccB_dmp; // Y-1 slice
    CudaDeviceMemoryPitched<TSimAcc, 2>* bestSimInYm1_dmpPtr  = &inout_volAxisAcc_dmp;   // best sim score along the Y axis for each Z value

    // Copy the first XZ plane (at Y=0) from 'in_volSim_dmp' into 'xzSliceForYm1_dmpPtr'
    volume_getVolumeXZSlice_kernel<TSimAcc, TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
        xzSliceForYm1_dmpPtr->getBuffer(),
        xzSliceForYm1_dmpPtr->getPitch(),
        in_volSim_dmp.getBuffer(),
        in_volSim_dmp.getBytesPaddedUpToDim(1),
        in_volSim_dmp.getBytesPaddedUpToDim(0),
        volDim_, 
        axisT_, 
        0 /* Y = 0 */ ); 

    // Set the first Z plane from 'out_volAgr_dmp' to 255
    volume_initVolumeYSlice_kernel<TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
        out_volAgr_dmp.getBuffer(),
        out_volAgr_dmp.getBytesPaddedUpToDim(1),
        out_volAgr_dmp.getBytesPaddedUpToDim(0),
        volDim_, 
        axisT_,
        0, 255);

    for(int iy = 1; iy < volDimY; ++iy)
    {
        const int y = invY ? volDimY - 1 - iy : iy;

        // For each column: compute the best score
        // Foreach x:
        //   bestSimInYm1[x] = min(d_xzSliceForY[1:height])
        volume_computeBestZInSlice_kernel<<<gridColZ, blockColZ, 0, stream>>>(
            xzSliceForYm1_dmpPtr->getBuffer(), 
            xzSliceForYm1_dmpPtr->getPitch(),
            bestSimInYm1_dmpPtr->getBuffer(),
            volDimX, volDimZ);

        // Copy the 'z' plane from 'in_volSim_dmp' into 'xzSliceForY'
        volume_getVolumeXZSlice_kernel<TSimAcc, TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
            xzSliceForY_dmpPtr->getBuffer(),
            xzSliceForY_dmpPtr->getPitch(),
            in_volSim_dmp.getBuffer(),
            in_volSim_dmp.getBytesPaddedUpToDim(1),
            in_volSim_dmp.getBytesPaddedUpToDim(0),
            volDim_, axisT_, y);

        volume_agregateCostVolumeAtXinSlices_kernel<<<gridVolSlide, blockVolSlide, 0, stream>>>(
            rcDeviceMipmapImage.getTextureObject(),
            (unsigned int)(rcLevelDim.x()),
            (unsigned int)(rcLevelDim.y()),
            rcMipmapLevel,
            xzSliceForY_dmpPtr->getBuffer(),   // inout: xzSliceForY
            xzSliceForY_dmpPtr->getPitch(),     
            xzSliceForYm1_dmpPtr->getBuffer(), // in:    xzSliceForYm1
            xzSliceForYm1_dmpPtr->getPitch(), 
            bestSimInYm1_dmpPtr->getBuffer(),  // in:    bestSimInYm1                        
            out_volAgr_dmp.getBuffer(), 
            out_volAgr_dmp.getBytesPaddedUpToDim(1), 
            out_volAgr_dmp.getBytesPaddedUpToDim(0), 
            volDim_, axisT_, 
            sgmParams.stepXY, 
            y, 
            sgmParams.p1, 
            sgmParams.p2Weighting,
            ySign, 
            filteringIndex,
            roi);

        std::swap(xzSliceForYm1_dmpPtr, xzSliceForY_dmpPtr);
    }
    
    // check cuda last error
    CHECK_CUDA_ERROR();
}

void cuda_volumeOptimize(CudaDeviceMemoryPitched<TSim, 3>& out_volSimFiltered_dmp,
                                  CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccA_dmp,
                                  CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccB_dmp,
                                  CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volAxisAcc_dmp,
                                  const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp, 
                                  const DeviceMipmapImage& rcDeviceMipmapImage,
                                  const SgmParams& sgmParams, 
                                  const int lastDepthIndex,
                                  const ROI& roi,
                                  cudaStream_t stream)
{
    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // update aggregation volume
    int npaths = 0;
    const auto updateAggrVolume = [&](const CudaSize<3>& axisT, bool invX)
    {
        cuda_volumeAggregatePath(out_volSimFiltered_dmp, 
                                 inout_volSliceAccA_dmp, 
                                 inout_volSliceAccB_dmp,
                                 inout_volAxisAcc_dmp,
                                 in_volSim_dmp, 
                                 rcDeviceMipmapImage,
                                 rcLevelDim,
                                 rcMipmapLevel,
                                 axisT, 
                                 sgmParams, 
                                 lastDepthIndex,
                                 npaths,
                                 invX, 
                                 roi,
                                 stream);
        npaths++;
    };

    // filtering is done on the last axis
    const std::map<char, CudaSize<3>> mapAxes = {
        {'X', {1, 0, 2}}, // XYZ -> YXZ
        {'Y', {0, 1, 2}}, // XYZ
    };

    for(char axis : sgmParams.filteringAxes)
    {
        const CudaSize<3>& axisT = mapAxes.at(axis);
        updateAggrVolume(axisT, false); // without transpose
        updateAggrVolume(axisT, true);  // with transpose of the last axis
    }
}

void cuda_volumeRetrieveBestDepth(CudaDeviceMemoryPitched<float2, 2>& out_sgmDepthThicknessMap_dmp,
                                           CudaDeviceMemoryPitched<float2, 2>& out_sgmDepthSimMap_dmp,
                                           const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp, 
                                           const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp, 
                                           const int rcDeviceCameraParamsId,
                                           const SgmParams& sgmParams, 
                                           const Range& depthRange,
                                           const ROI& roi, 
                                           cudaStream_t stream)
{
    // constant kernel inputs
    const int scaleStep = sgmParams.scale * sgmParams.stepXY;
    const float thicknessMultFactor = 1.f + float(sgmParams.depthThicknessInflate);
    const float maxSimilarity = float(sgmParams.maxSimilarity) * 254.f; // convert from (0, 1) to (0, 254)

    // kernel launch parameters
    const dim3 block = getMaxPotentialBlockSize(volume_retrieveBestDepth_kernel);
    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);
    
    // kernel execution
    volume_retrieveBestDepth_kernel<<<grid, block, 0, stream>>>(
        out_sgmDepthThicknessMap_dmp.getBuffer(),
        out_sgmDepthThicknessMap_dmp.getBytesPaddedUpToDim(0),
        out_sgmDepthSimMap_dmp.getBuffer(),
        out_sgmDepthSimMap_dmp.getBytesPaddedUpToDim(0),
        in_depths_dmp.getBuffer(),
        in_depths_dmp.getBytesPaddedUpToDim(0),
        in_volSim_dmp.getBuffer(),
        in_volSim_dmp.getBytesPaddedUpToDim(1),
        in_volSim_dmp.getBytesPaddedUpToDim(0),
        rcDeviceCameraParamsId,
        int(in_volSim_dmp.getSize().z()),
        scaleStep,
        thicknessMultFactor,
        maxSimilarity,
        depthRange,
        roi);

    // check cuda last error
    CHECK_CUDA_ERROR();
}

extern void cuda_volumeRefineBestDepth(CudaDeviceMemoryPitched<float2, 2>& out_refineDepthSimMap_dmp,
                                       const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                       const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volSim_dmp, 
                                       const RefineParams& refineParams, 
                                       const ROI& roi, 
                                       cudaStream_t stream)
{
    // constant kernel inputs
    const int halfNbSamples = refineParams.nbSubsamples * refineParams.halfNbDepths;
    const float twoTimesSigmaPowerTwo = float(2.0 * refineParams.sigma * refineParams.sigma);

    // kernel launch parameters
    const dim3 block = getMaxPotentialBlockSize(volume_refineBestDepth_kernel);
    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);

    // kernel execution
    volume_refineBestDepth_kernel<<<grid, block, 0, stream>>>(
        out_refineDepthSimMap_dmp.getBuffer(),
        out_refineDepthSimMap_dmp.getBytesPaddedUpToDim(0),
        in_sgmDepthPixSizeMap_dmp.getBuffer(),
        in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0),
        in_volSim_dmp.getBuffer(),
        in_volSim_dmp.getBytesPaddedUpToDim(1),
        in_volSim_dmp.getBytesPaddedUpToDim(0),
        int(in_volSim_dmp.getSize().z()),
        refineParams.nbSubsamples,  // number of samples between two depths
        halfNbSamples,              // number of samples (in front and behind mid depth)
        refineParams.halfNbDepths,  // number of depths  (in front and behind mid depth)
        twoTimesSigmaPowerTwo,
        roi);

    // check cuda last error
    CHECK_CUDA_ERROR();
}

} // namespace depthMap

