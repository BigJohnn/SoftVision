// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
#import <depthMap/gpu/host/ComputePipeline.hpp>
#import <depthMap/gpu/host/DeviceTexture.hpp>

#include "deviceSimilarityVolume.hpp"

#include <mvsData/ROI_d.hpp>
//#include <depthMap/gpu/host/divUp.hpp>

#include <map>

#define SOFTVISION_DEBUG

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
//template<class T>
//dim3 getMaxPotentialBlockSize(T kernelFuction)
//{
//    const dim3 defaultBlock(32, 1, 1); // minimal default settings
//
//    int recommendedMinGridSize;
//    int recommendedBlockSize;
//
//    cudaError_t err;
//    err = cudaOccupancyMaxPotentialBlockSize(&recommendedMinGridSize,
//                                             &recommendedBlockSize,
//                                             kernelFuction,
//                                             0, // dynamic shared mem size: none used
//                                             0); // no block size limit, 1 thread OK
//
//    if(err != cudaSuccess)
//    {
//        ALICEVISION_LOG_WARNING( "cudaOccupancyMaxPotentialBlockSize failed, using default block settings.");
//        return defaultBlock;
//    }
//
//    if(recommendedBlockSize > 32)
//    {
//        const dim3 recommendedBlock(32, divUp(recommendedBlockSize, 32), 1);
//        return recommendedBlock;
//    }
//
//    return defaultBlock;
//}

void volumeInitialize(DeviceBuffer* inout_volume_dmp, TSim value)
{
    // get input/output volume dimensions
    MTLSize volDim = [inout_volume_dmp getSize];

    // kernel launch parameters
    MTLSize block = MTLSizeMake(32, 4, 1);
//    MTLSize grid = MTLSizeMake((volDim.width + block.width - 1)/block.width, (volDim.height+block.height-1)/block.height, volDim.depth);
    MTLSize threads = MTLSizeMake(volDim.width, volDim.height, volDim.depth);
//    const dim3 block(32, 4, 1);
//    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());

    // kernel execution
    
//    [inout_volume_dmp allocate:volDim elemSizeInBytes:sizeof(float)];
    NSArray* args = @[
        [inout_volume_dmp getBuffer],
        @([inout_volume_dmp getBytesUpToDim:1]),
        @([inout_volume_dmp getBytesUpToDim:0]),
        @((unsigned)volDim.width),
        @((unsigned)volDim.height),
        @((unsigned char)value)
    ];
    ComputePipeline* pipeline = [ComputePipeline createPipeline];

    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_init_kernel" Args:args];

//    auto* p = [inout_volume_dmp getBufferPtr];
//    NSLog(@"inout_volume_dmp addr==%p", p);
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

void volumeInitialize(DeviceBuffer* inout_volume_dmp, TSimRefine value)
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
    MTLSize threads = MTLSizeMake(volDim.width, volDim.height, volDim.depth);

    // kernel execution
    NSArray* args = @[
        [inout_volume_dmp getBuffer], //TODO: check: TSimRefine是否使用half
        @([inout_volume_dmp getBytesUpToDim:1]), //1024*256
        @([inout_volume_dmp getBytesUpToDim:0]), // 1024
        @((unsigned)volDim.width),
        @((unsigned)volDim.height),
        @(value)
    ];
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_init_kernel_refine" Args:args];
    
//    DeviceTexture* texture = [inout_volume_dmp getDebugTexture];
//    NSLog(@"volumeInitialize debug texture");
//    volume_init_kernel<TSimRefine><<<grid, block, 0, stream>>>(
//        inout_volume_dmp.getBuffer(),
//        inout_volume_dmp.getBytesPaddedUpToDim(1),
//        inout_volume_dmp.getBytesPaddedUpToDim(0),
//        (unsigned int)(volDim.x()),
//        (unsigned int)(volDim.y()),
//        value);
}

void cuda_volumeAdd(DeviceBuffer* inout_volume_dmp,
                             DeviceBuffer* in_volume_dmp)
{
    // get input/output volume dimensions
//    const CudaSize<3>& volDim = inout_volume_dmp.getSize();
//
//    // kernel launch parameters
//    const dim3 block(32, 4, 1);
//    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());
//
//    // kernel execution
//    volume_add_kernel<<<grid, block, 0, stream>>>(
//        inout_volume_dmp.getBuffer(),
//        inout_volume_dmp.getBytesPaddedUpToDim(1),
//        inout_volume_dmp.getBytesPaddedUpToDim(0),
//        in_volume_dmp.getBuffer(),
//        in_volume_dmp.getBytesPaddedUpToDim(1),
//        in_volume_dmp.getBytesPaddedUpToDim(0),
//        (unsigned int)(volDim.x()),
//        (unsigned int)(volDim.y()));
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void volumeUpdateUninitializedSimilarity(DeviceBuffer* in_volBestSim_dmp,
                                                       DeviceBuffer* inout_volSecBestSim_dmp)
{
//    assert(in_volBestSim_dmp.getSize() == inout_volSecBestSim_dmp.getSize());
//
    // get input/output volume dimensions
    const MTLSize& volDim = [inout_volSecBestSim_dmp getSize];

    // kernel launch parameters
//    const dim3 block = getMaxPotentialBlockSize(volume_updateUninitialized_kernel);
//    const dim3 grid(divUp(volDim.x(), block.x), divUp(volDim.y(), block.y), volDim.z());
    
    const MTLSize& block = MTLSizeMake(8, 8, 1);
    const MTLSize& threads = volDim;

    // kernel execution
    NSArray* args = @[
                [inout_volSecBestSim_dmp getBuffer],
                @([inout_volSecBestSim_dmp getBytesUpToDim:1]),
                @([inout_volSecBestSim_dmp getBytesUpToDim:0]),
        
                [in_volBestSim_dmp getBuffer],
                @([in_volBestSim_dmp getBytesUpToDim:1]),
                @([in_volBestSim_dmp getBytesUpToDim:0]),
        
                @((unsigned int)(volDim.width)),
                @((unsigned int)(volDim.height))
    ];
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_updateUninitialized_kernel" Args:args];
    
    
    
//    volume_updateUninitialized_kernel<<<grid, block, 0, stream>>>(
//        inout_volSecBestSim_dmp.getBuffer(),
//        inout_volSecBestSim_dmp.getBytesPaddedUpToDim(1),
//        inout_volSecBestSim_dmp.getBytesPaddedUpToDim(0),
//        in_volBestSim_dmp.getBuffer(),
//        in_volBestSim_dmp.getBytesPaddedUpToDim(1),
//        in_volBestSim_dmp.getBytesPaddedUpToDim(0),
//        (unsigned int)(volDim.x()),
//        (unsigned int)(volDim.y()));
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void volumeComputeSimilarity(DeviceBuffer* out_volBestSim_dmp,
                              DeviceBuffer* out_volSecBestSim_dmp,
                               DeviceBuffer* in_depths_dmp,
                             DeviceCameraParams const& rcDeviceCameraParams,
                             DeviceCameraParams const& tcDeviceCameraParams,
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
    MTLSize threads = MTLSizeMake(roi.width(), roi.height(), depthRange.size());

    // kernel execution
    
//    [inout_volume_dmp allocate:volDim elemSizeInBytes:sizeof(float)];
    simd_uint2 depthRange_d = simd_make_uint2(depthRange.begin, depthRange.end);
//    depthRange_d.begin = depthRange.begin;
//    depthRange_d.end = depthRange.end;
    
    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    NSArray* args = @[
        [out_volBestSim_dmp getBuffer], //256x256x1500 uchar
        @([out_volBestSim_dmp getBytesUpToDim:1]), //256*256
        @([out_volBestSim_dmp getBytesUpToDim:0]), // 256
        [out_volSecBestSim_dmp getBuffer],
        @([out_volSecBestSim_dmp getBytesUpToDim:1]), //256*256
        @([out_volSecBestSim_dmp getBytesUpToDim:0]), // 256
        [in_depths_dmp getBuffer],
        @([in_depths_dmp getBytesUpToDim:0]), // 6000
        [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
        [NSData dataWithBytes:&tcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
        rcDeviceMipmapImage.getTextureObject(),
        tcDeviceMipmapImage.getTextureObject(),
        @((unsigned)rcLevelDim.width),//180
        @((unsigned)rcLevelDim.height),//320
        @((unsigned)tcLevelDim.width),//180
        @((unsigned)tcLevelDim.height),//320
        @(rcMipmapLevel),//1
        @(sgmParams.stepXY),
        @(sgmParams.wsh),
        
        @(1.f / float(sgmParams.gammaC)),
        @(1.f / float(sgmParams.gammaP)),
        
        @(sgmParams.useConsistentScale), //false
        @(sgmParams.useCustomPatchPattern), //false
        
        [NSData dataWithBytes:&depthRange_d length:sizeof(depthRange_d)],
        [NSData dataWithBytes:&roi_d length:sizeof(ROI_d)]
    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];

    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_computeSimilarity_kernel" Args:args];

    
//    id<MTLTexture> texture_d1 = [out_volBestSim_dmp getDebugTexture:depthRange_d[0]];
//    id<MTLTexture> texture_d2 = [out_volSecBestSim_dmp getDebugTexture:depthRange_d[0]];
//    
//    NSLog(@"debug pause");//
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

extern void volumeRefineSimilarity(DeviceBuffer* inout_volSim_dmp,
                                        DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                        DeviceBuffer* in_sgmNormalMap_dmpPtr,
                                   DeviceCameraParams const& rcDeviceCameraParams,
                                   DeviceCameraParams const& tcDeviceCameraParams,
                                        const DeviceMipmapImage& rcDeviceMipmapImage,
                                        const DeviceMipmapImage& tcDeviceMipmapImage,
                                        const RefineParams& refineParams, 
                                        const Range& depthRange,
                                        const ROI& roi)
{
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const MTLSize rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
    const MTLSize tcLevelDim = tcDeviceMipmapImage.getDimensions(refineParams.scale);

    // kernel launch parameters
//    const dim3 block = getMaxPotentialBlockSize(volume_refineSimilarity_kernel);
//    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), depthRange.size());
    const MTLSize block = MTLSizeMake(16, 16, 1);
    const MTLSize threads = MTLSizeMake(roi.width(), roi.height(), depthRange.size());
    
    simd_uint2 depthRange_d = simd_make_uint2(depthRange.begin, depthRange.end);
//    depthRange_d.begin = depthRange.begin;
//    depthRange_d.end = depthRange.end;
    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    NSArray* args = @[
                [inout_volSim_dmp getBuffer],
                @([inout_volSim_dmp getBytesUpToDim:1]),
                @([inout_volSim_dmp getBytesUpToDim:0]),
                [in_sgmDepthPixSizeMap_dmp getBuffer],
                @([in_sgmDepthPixSizeMap_dmp getBytesUpToDim:0]),
                (in_sgmNormalMap_dmpPtr == nil) ? [NSNull null] : [in_sgmNormalMap_dmpPtr getBuffer],
                (in_sgmNormalMap_dmpPtr == nil) ? @(0) : @([in_sgmNormalMap_dmpPtr getBytesUpToDim:0]),
                [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                [NSData dataWithBytes:&tcDeviceCameraParams length:sizeof(tcDeviceCameraParams)],
                rcDeviceMipmapImage.getTextureObject(),
                tcDeviceMipmapImage.getTextureObject(),
                @((unsigned int)(rcLevelDim.width)),
                @((unsigned int)(rcLevelDim.height)),
                @((unsigned int)(tcLevelDim.width)),
                @((unsigned int)(tcLevelDim.height)),
                @(rcMipmapLevel),
                @(int([inout_volSim_dmp getSize].depth)),
                @(refineParams.stepXY),
                @(refineParams.wsh),
                @((1.f / float(refineParams.gammaC))), // inverted gammaC
                @((1.f / float(refineParams.gammaP))), // inverted gammaP
                @(refineParams.useConsistentScale),
                @(refineParams.useCustomPatchPattern),
                [NSData dataWithBytes:&depthRange_d length:sizeof(depthRange_d)],
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]

    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_refineSimilarity_kernel" Args:args];
   
#ifdef SOFTVISION_DEBUG
//    DeviceTexture* texture = [inout_volSim_dmp getDebugTexture];
//    NSLog(@"...");
#endif
    // kernel execution
//    volume_refineSimilarity_kernel<<<grid, block, 0, stream>>>(
//        inout_volSim_dmp.getBuffer(),
//        inout_volSim_dmp.getBytesPaddedUpToDim(1),
//        inout_volSim_dmp.getBytesPaddedUpToDim(0),
//        in_sgmDepthPixSizeMap_dmp.getBuffer(),
//        in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0),
//        (in_sgmNormalMap_dmpPtr == nullptr) ? nullptr : in_sgmNormalMap_dmpPtr->getBuffer(),
//        (in_sgmNormalMap_dmpPtr == nullptr) ? 0 : in_sgmNormalMap_dmpPtr->getBytesPaddedUpToDim(0),
//        rcDeviceCameraParamsId,
//        tcDeviceCameraParamsId,
//        rcDeviceMipmapImage.getTextureObject(),
//        tcDeviceMipmapImage.getTextureObject(),
//        (unsigned int)(rcLevelDim.x()),
//        (unsigned int)(rcLevelDim.y()),
//        (unsigned int)(tcLevelDim.x()),
//        (unsigned int)(tcLevelDim.y()),
//        rcMipmapLevel,
//        int(inout_volSim_dmp.getSize().z()),
//        refineParams.stepXY,
//        refineParams.wsh,
//        (1.f / float(refineParams.gammaC)), // inverted gammaC
//        (1.f / float(refineParams.gammaP)), // inverted gammaP
//        refineParams.useConsistentScale,
//        refineParams.useCustomPatchPattern,
//        depthRange,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}


void volumeAggregatePath(DeviceBuffer* out_volAgr_dmp,
                           DeviceBuffer* inout_volSliceAccA_dmp,
                           DeviceBuffer* inout_volSliceAccB_dmp,
                           DeviceBuffer* inout_volAxisAcc_dmp,
                           DeviceBuffer* in_volSim_dmp,
                           const DeviceMipmapImage& rcDeviceMipmapImage,
                           MTLSize const& rcLevelDim,
                           const float rcMipmapLevel,
                           bool axisT,
                           const SgmParams& sgmParams,
                           const int lastDepthIndex,
                           const int filteringIndex,
                           const bool invY,
                           const ROI& roi)
{
    MTLSize volDim = [in_volSim_dmp getSize];
    volDim.depth = lastDepthIndex; // override volume depth, use rc depth list last index

    size_t volDimX = volDim.width;
    size_t volDimY = volDim.height;
    size_t volDimZ = volDim.depth;
    if(axisT) {
        std::swap(volDimX, volDimY);
    }

    simd_int3 volDim_ = simd_make_int3(volDim.width, volDim.height, volDim.depth);
    simd_int3 axisT_ = axisT ? simd_make_int3(1,0,2) : simd_make_int3(0,1,2);
    const int ySign = (invY ? -1 : 1);

    // setup block and grid
    const int blockSize = 8;
    MTLSize blockVolXZ = MTLSizeMake(blockSize, blockSize, 1);
    MTLSize gridVolXZ = MTLSizeMake(volDimX, volDimZ, 1);

    const int blockSizeL = 64;
    MTLSize blockColZ = MTLSizeMake(blockSizeL, 1, 1);
    MTLSize gridColZ = MTLSizeMake(volDimX, 1, 1);

    MTLSize blockVolSlide = MTLSizeMake(blockSizeL, 1, 1);
    MTLSize gridVolSlide = MTLSizeMake(volDimX, volDimZ, 1);

    DeviceBuffer* xzSliceForY_dmpPtr   = inout_volSliceAccA_dmp; // Y slice
    DeviceBuffer* xzSliceForYm1_dmpPtr = inout_volSliceAccB_dmp; // Y-1 slice
    DeviceBuffer* bestSimInYm1_dmpPtr  = inout_volAxisAcc_dmp;   // best sim score along the Y axis for each Z value

    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    
    // Copy the first XZ plane (at Y=0) from 'in_volSim_dmp' into 'xzSliceForYm1_dmpPtr'
    {
        NSArray* args = @[
            [xzSliceForYm1_dmpPtr getBuffer],
            @([xzSliceForYm1_dmpPtr getBytesUpToDim:0]), // getPitch
            [in_volSim_dmp getBuffer],
            @([in_volSim_dmp getBytesUpToDim:1]), //256*256
            @([in_volSim_dmp getBytesUpToDim:0]), // 256
            [NSData dataWithBytes:&volDim_ length:sizeof(volDim_)],
            [NSData dataWithBytes:&axisT_ length:sizeof(axisT_)],
            @(0) // Y = 0
        ];
        
        [pipeline Exec:gridVolXZ ThreadgroupSize:blockVolXZ KernelFuncName:@"depthMap::volume_getVolumeXZSlice_kernel" Args:args];
        
//        id<MTLTexture> texture_d1 = [xzSliceForYm1_dmpPtr getDebugTexture:0];
//        
//        NSLog(@"debug pause");//
    }
    
    
    
//    // Copy the first XZ plane (at Y=0) from 'in_volSim_dmp' into 'xzSliceForYm1_dmpPtr'
//    volume_getVolumeXZSlice_kernel<TSimAcc, TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
//        xzSliceForYm1_dmpPtr->getBuffer(),
//        xzSliceForYm1_dmpPtr->getPitch(),
//        in_volSim_dmp.getBuffer(),
//        in_volSim_dmp.getBytesPaddedUpToDim(1),
//        in_volSim_dmp.getBytesPaddedUpToDim(0),
//        volDim_,
//        axisT_,
//        0 /* Y = 0 */ );

    // Set the first Z plane from 'out_volAgr_dmp' to 255
    {
        NSArray* args = @[
            [out_volAgr_dmp getBuffer],
            @([out_volAgr_dmp getBytesUpToDim:1]), //wbytes * h
            @([out_volAgr_dmp getBytesUpToDim:0]), // wbytes
            [NSData dataWithBytes:&volDim_ length:sizeof(volDim_)],
            [NSData dataWithBytes:&axisT_ length:sizeof(axisT_)],
            @(0),
            @((TSim)255) //TSim
        ];
        
        [pipeline Exec:gridVolXZ ThreadgroupSize:blockVolXZ KernelFuncName:@"depthMap::volume_initVolumeYSlice_kernel" Args:args];
        
//        id<MTLTexture> texture_d1 = [out_volAgr_dmp getDebugTexture:27];
//        
//        NSLog(@"debug pause");//
    }
    
    
//    volume_initVolumeYSlice_kernel<TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
//        out_volAgr_dmp.getBuffer(),
//        out_volAgr_dmp.getBytesPaddedUpToDim(1),
//        out_volAgr_dmp.getBytesPaddedUpToDim(0),
//        volDim_,
//        axisT_,
//        0, 255);
    
    
    for(int iy = 1; iy < volDimY; ++iy)
    {
        const int y = invY ? volDimY - 1 - iy : iy;

        
        // For each column: compute the best score
        // Foreach x:
        //   bestSimInYm1[x] = min(d_xzSliceForY[1:height])
        {
            NSArray* args = @[
                [xzSliceForYm1_dmpPtr getBuffer],
                @([xzSliceForYm1_dmpPtr getBytesUpToDim:0]), // wbytes
                [bestSimInYm1_dmpPtr getBuffer],
                @(volDimX),
                @(volDimZ)
            ];
            
            [pipeline Exec:gridColZ ThreadgroupSize:blockColZ KernelFuncName:@"depthMap::volume_computeBestZInSlice_kernel" Args:args];
            
//            id<MTLTexture> texture_d1 = [xzSliceForYm1_dmpPtr getDebugTexture:0];
//            
//            NSLog(@"debug pause");//
        }
//        volume_computeBestZInSlice_kernel<<<gridColZ, blockColZ, 0, stream>>>(
//            xzSliceForYm1_dmpPtr->getBuffer(),
//            xzSliceForYm1_dmpPtr->getPitch(),
//            bestSimInYm1_dmpPtr->getBuffer(),
//            volDimX, volDimZ);

        // Copy the 'z' plane from 'in_volSim_dmp' into 'xzSliceForY'
        {
            NSArray* args = @[
                [xzSliceForY_dmpPtr getBuffer],
                @([xzSliceForY_dmpPtr getBytesUpToDim:0]), // wbytes
                [in_volSim_dmp getBuffer],
                @([in_volSim_dmp getBytesUpToDim:1]), //wbytes * h
                @([in_volSim_dmp getBytesUpToDim:0]), // wbytes
                [NSData dataWithBytes:&volDim_ length:sizeof(volDim_)],
                [NSData dataWithBytes:&axisT_ length:sizeof(axisT_)],
                @(y)
            ];
            
            [pipeline Exec:gridVolXZ ThreadgroupSize:blockVolXZ KernelFuncName:@"depthMap::volume_getVolumeXZSlice_kernel" Args:args];
            
//            id<MTLTexture> texture_d1 = [xzSliceForY_dmpPtr getDebugTexture:0];
//            
//            NSLog(@"debug pause");//
        }
//        volume_getVolumeXZSlice_kernel<TSimAcc, TSim><<<gridVolXZ, blockVolXZ, 0, stream>>>(
//            xzSliceForY_dmpPtr->getBuffer(),
//            xzSliceForY_dmpPtr->getPitch(),
//            in_volSim_dmp.getBuffer(),
//            in_volSim_dmp.getBytesPaddedUpToDim(1),
//            in_volSim_dmp.getBytesPaddedUpToDim(0),
//            volDim_, axisT_, y);

        {
            ROI_d roi_d;//(roi.x.begin, roi.y.begin, roi.x.end, roi.y.end);
            roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
            roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
            NSArray* args = @[
                rcDeviceMipmapImage.getTextureObject(),
                @(rcLevelDim.width),
                @(rcLevelDim.height),
                @(rcMipmapLevel),
                [xzSliceForY_dmpPtr getBuffer],// inout: xzSliceForY
                @([xzSliceForY_dmpPtr getBytesUpToDim:0]), // wbytes
                [xzSliceForYm1_dmpPtr getBuffer],// in:    xzSliceForYm1
                @([xzSliceForYm1_dmpPtr getBytesUpToDim:0]), // wbytes
                [bestSimInYm1_dmpPtr getBuffer],// in:    bestSimInYm1
                [out_volAgr_dmp getBuffer],// in:    bestSimInYm1
                @([out_volAgr_dmp getBytesUpToDim:1]), //wbytes * h
                @([out_volAgr_dmp getBytesUpToDim:0]), // wbytes
                [NSData dataWithBytes:&volDim_ length:sizeof(volDim_)],
                [NSData dataWithBytes:&axisT_ length:sizeof(axisT_)],
                @(sgmParams.stepXY*1.0f),
                @(y),
                @(sgmParams.p1),
                @(sgmParams.p2Weighting),
                @(ySign),
                @(filteringIndex),
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
            ];
            
            [pipeline Exec:gridVolSlide ThreadgroupSize:blockVolSlide KernelFuncName:@"depthMap::volume_agregateCostVolumeAtXinSlices_kernel" Args:args];

//            {
//                id<MTLTexture> texture_d = [xzSliceForY_dmpPtr getDebugTexture];
//                
//                
//                id<MTLTexture> texture_d1 = [out_volAgr_dmp getDebugTexture:y];
//                NSLog(@"xxx");
//            }
            
            
            
        }
        
//        volume_agregateCostVolumeAtXinSlices_kernel<<<gridVolSlide, blockVolSlide, 0, stream>>>(
//            rcDeviceMipmapImage.getTextureObject(),
//            (unsigned int)(rcLevelDim.x()),
//            (unsigned int)(rcLevelDim.y()),
//            rcMipmapLevel,
//            xzSliceForY_dmpPtr->getBuffer(),   // inout: xzSliceForY
//            xzSliceForY_dmpPtr->getPitch(),
//            xzSliceForYm1_dmpPtr->getBuffer(), // in:    xzSliceForYm1
//            xzSliceForYm1_dmpPtr->getPitch(),
//            bestSimInYm1_dmpPtr->getBuffer(),  // in:    bestSimInYm1
//            out_volAgr_dmp.getBuffer(),
//            out_volAgr_dmp.getBytesPaddedUpToDim(1),
//            out_volAgr_dmp.getBytesPaddedUpToDim(0),
//            volDim_, axisT_,
//            sgmParams.stepXY,
//            y,
//            sgmParams.p1,
//            sgmParams.p2Weighting,
//            ySign,
//            filteringIndex,
//            roi);

        std::swap(xzSliceForYm1_dmpPtr, xzSliceForY_dmpPtr);
    }
    
    {
        
//        id<MTLTexture> texture_dx = [xzSliceForY_dmpPtr getDebugTexture];
//        
//        id<MTLTexture> texture_d = [out_volAgr_dmp getDebugTexture:0];
//        id<MTLTexture> texture_d1 = [out_volAgr_dmp getDebugTexture:1];
//        id<MTLTexture> texture_d2 = [out_volAgr_dmp getDebugTexture:2];
//        id<MTLTexture> texture_d3 = [out_volAgr_dmp getDebugTexture:3];
//        NSLog(@"xxx");
    }
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void volumeOptimize(DeviceBuffer* out_volSimFiltered_dmp,
                                  DeviceBuffer* inout_volSliceAccA_dmp,
                                  DeviceBuffer* inout_volSliceAccB_dmp,
                                  DeviceBuffer* inout_volAxisAcc_dmp,
                                  DeviceBuffer* in_volSim_dmp, 
                                  const DeviceMipmapImage& rcDeviceMipmapImage,
                                  const SgmParams& sgmParams, 
                                  const int lastDepthIndex,
                                  const ROI& roi)
{
    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale); //check:
    MTLSize rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // update aggregation volume
    int npaths = 0;
    const auto updateAggrVolume = [&](bool axisT, bool invX)
    {
        volumeAggregatePath(out_volSimFiltered_dmp,
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
                                 roi);
        npaths++;
    };

    // filtering is done on the last axis
    const std::map<char, bool> mapAxes = {
        {'X', true}, // XYZ -> YXZ
        {'Y', false}, // XYZ
    };

    for(char axis : sgmParams.filteringAxes)
    {
        bool axisT = mapAxes.at(axis);
        updateAggrVolume(axisT, false); // without transpose
        updateAggrVolume(axisT, true);  // with transpose of the last axis
    }
}

void volumeRetrieveBestDepth(DeviceBuffer* out_sgmDepthThicknessMap_dmp,
                                           DeviceBuffer* out_sgmDepthSimMap_dmp,
                                           DeviceBuffer* in_depths_dmp, 
                                           DeviceBuffer* in_volSim_dmp, 
                                           const DeviceCameraParams& rcDeviceCameraParams,
                                           const SgmParams& sgmParams,
                                           const Range& depthRange,
                                           const ROI& roi)
{
    // constant kernel inputs
    const int scaleStep = sgmParams.scale * sgmParams.stepXY;
    const float thicknessMultFactor = 1.f + float(sgmParams.depthThicknessInflate);
    const float maxSimilarity = float(sgmParams.maxSimilarity) * 254.f; // convert from (0, 1) to (0, 254)

    // kernel launch parameters
//    const dim3 block = getMaxPotentialBlockSize(volume_retrieveBestDepth_kernel);
//    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);
    const MTLSize& block = MTLSizeMake(8, 8, 1);
    const MTLSize& threads = MTLSizeMake(roi.width(), roi.height(), 1);

    simd_uint2 depthRange_d = simd_make_uint2(depthRange.begin, depthRange.end);
    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    // kernel execution
    NSArray* args = @[
                [out_sgmDepthThicknessMap_dmp getBuffer],
                @([out_sgmDepthThicknessMap_dmp getBytesUpToDim:0]),
                (out_sgmDepthSimMap_dmp == nil) ? [NSNull null] : [out_sgmDepthSimMap_dmp getBuffer],
                (out_sgmDepthSimMap_dmp == nil) ? @(0) : @([out_sgmDepthSimMap_dmp getBytesUpToDim:0]),
                [in_depths_dmp getBuffer],
                @([in_depths_dmp getBytesUpToDim:0]),
                [in_volSim_dmp getBuffer],
                @([in_volSim_dmp getBytesUpToDim:1]),
                @([in_volSim_dmp getBytesUpToDim:0]),
                [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                @((int)[in_volSim_dmp getSize].depth),
                @(scaleStep),
                @(thicknessMultFactor),
                @(maxSimilarity),
                [NSData dataWithBytes:&depthRange_d length:sizeof(depthRange_d)],
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]

    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_retrieveBestDepth_kernel" Args:args];
    
//    id<MTLTexture> texture_d3 = [in_depths_dmp getDebugTexture:0];
//    id<MTLTexture> texture_d4 = [in_volSim_dmp getDebugTexture:10];
//    id<MTLTexture> texture_d1 = [out_sgmDepthThicknessMap_dmp getDebugTexture:0];
//    id<MTLTexture> texture_d2 = [out_sgmDepthSimMap_dmp getDebugTexture:0];
//    NSLog(@"xxx");
//    volume_retrieveBestDepth_kernel<<<grid, block, 0, stream>>>(
//        out_sgmDepthThicknessMap_dmp.getBuffer(),
//        out_sgmDepthThicknessMap_dmp.getBytesPaddedUpToDim(0),
//        out_sgmDepthSimMap_dmp.getBuffer(),
//        out_sgmDepthSimMap_dmp.getBytesPaddedUpToDim(0),
//        in_depths_dmp.getBuffer(),
//        in_depths_dmp.getBytesPaddedUpToDim(0),
//        in_volSim_dmp.getBuffer(),
//        in_volSim_dmp.getBytesPaddedUpToDim(1),
//        in_volSim_dmp.getBytesPaddedUpToDim(0),
//        rcDeviceCameraParamsId,
//        int(in_volSim_dmp.getSize().z()),
//        scaleStep,
//        thicknessMultFactor,
//        maxSimilarity,
//        depthRange,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

extern void volumeRefineBestDepth(DeviceBuffer* out_refineDepthSimMap_dmp,
                                       DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                       DeviceBuffer* in_volSim_dmp,
                                       const RefineParams& refineParams, 
                                       const ROI& roi)
{
    // constant kernel inputs
    const int halfNbSamples = refineParams.nbSubsamples * refineParams.halfNbDepths;
    const float twoTimesSigmaPowerTwo = float(2.0 * refineParams.sigma * refineParams.sigma);

    // kernel launch parameters
//    const dim3 block = getMaxPotentialBlockSize(volume_refineBestDepth_kernel);
//    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);
    MTLSize const& block = MTLSizeMake(16, 16, 1);
    MTLSize const& threads = MTLSizeMake(roi.width(), roi.height(), 1);
    
    // kernel execution
    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    NSArray* args = @[
                [out_refineDepthSimMap_dmp getBuffer],
                @([out_refineDepthSimMap_dmp getBytesUpToDim:0]),
                [in_sgmDepthPixSizeMap_dmp getBuffer],
                @([in_sgmDepthPixSizeMap_dmp getBytesUpToDim:0]),
                [in_volSim_dmp getBuffer],
                @([in_volSim_dmp getBytesUpToDim:1]),
                @([in_volSim_dmp getBytesUpToDim:0]),
                @(int([in_volSim_dmp getSize].depth)),
                @(refineParams.nbSubsamples),  // number of samples between two depths
                @(halfNbSamples),              // number of samples (in front and behind mid depth)
                @(refineParams.halfNbDepths),  // number of depths  (in front and behind mid depth)
                @(twoTimesSigmaPowerTwo),
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]

    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::volume_refineBestDepth_kernel" Args:args];
    
#ifdef SOFTVISION_DEBUG
    
//    DeviceTexture* texture_in = [in_volSim_dmp getDebugTexture];
//    DeviceTexture* texture_depth_pixsize_in = [in_sgmDepthPixSizeMap_dmp getDebugTexture];
//    DeviceTexture* texture = [out_refineDepthSimMap_dmp getDebugTexture];
//    NSLog(@"volumeRefineBestDepth...");
#endif
//    volume_refineBestDepth_kernel<<<grid, block, 0, stream>>>(
//        out_refineDepthSimMap_dmp.getBuffer(),
//        out_refineDepthSimMap_dmp.getBytesPaddedUpToDim(0),
//        in_sgmDepthPixSizeMap_dmp.getBuffer(),
//        in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0),
//        in_volSim_dmp.getBuffer(),
//        in_volSim_dmp.getBytesPaddedUpToDim(1),
//        in_volSim_dmp.getBytesPaddedUpToDim(0),
//        int(in_volSim_dmp.getSize().z()),
//        refineParams.nbSubsamples,  // number of samples between two depths
//        halfNbSamples,              // number of samples (in front and behind mid depth)
//        refineParams.halfNbDepths,  // number of depths  (in front and behind mid depth)
//        twoTimesSigmaPowerTwo,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

} // namespace depthMap

