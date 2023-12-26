// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
//#import <Metal/Metal.h>

#import <depthMap/gpu/host/ComputePipeline.hpp>

#include "deviceDepthSimilarityMap.hpp"
//#include "deviceDepthSimilarityMapKernels.cuh"

#include <depthMap/gpu/host/divUp.hpp>
#include <depthMap/gpu/device/DeviceCameraParams.hpp>

#include <utility>
#include <mvsData/ROI_d.hpp>

#include <simd/simd.h>


namespace depthMap {

void depthSimMapCopyDepthOnly(DeviceBuffer* out_depthSimMap_dmp,
                                            DeviceBuffer* in_depthSimMap_dmp,
                                            float defaultSim)
{
    // get output map dimensions
//    const CudaSize<2>& depthSimMapDim = out_depthSimMap_dmp.getSize();
    
    MTLSize depthSimMapDim = [out_depthSimMap_dmp getSize];
    // kernel launch parameters
//    const int blockSize = 16;
//    const dim3 block(blockSize, blockSize, 1);
//    const dim3 grid(divUp(depthSimMapDim.x(), blockSize), divUp(depthSimMapDim.y(), blockSize), 1);
    
    NSUInteger threadGroupSize = 16;
//    MTLSize gridSize = MTLSizeMake(divUp(depthSimMapDim.x(), threadGroupSize), divUp(depthSimMapDim.y(), threadGroupSize), 1);
    MTLSize threadsSize = MTLSizeMake(depthSimMapDim.width, depthSimMapDim.height, 1);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, threadGroupSize, 1);

    // kernel execution
    NSArray* args = @[
        [out_depthSimMap_dmp getBuffer],
        @([out_depthSimMap_dmp getBytesUpToDim:0]),
        [in_depthSimMap_dmp getBuffer],
        @([in_depthSimMap_dmp getBytesUpToDim:0]),
        @(depthSimMapDim.width),
        @(depthSimMapDim.height),
        @(defaultSim)
    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threadsSize ThreadgroupSize:threadgroupSize KernelFuncName:@"depthMap::depthSimMapCopyDepthOnly_kernel" Args:args];
}

void normalMapUpscale(DeviceBuffer* out_upscaledMap_dmp,
                                    DeviceBuffer* in_map_dmp,
                                    const ROI& roi)
{
    // compute upscale ratio
    const MTLSize& out_mapDim = [out_upscaledMap_dmp getSize];
    const MTLSize& in_mapDim = [in_map_dmp getSize];
    const float ratio = float(in_mapDim.width) / float(out_mapDim.width);

    // kernel launch parameters
    const int blockSize = 16;
    const MTLSize block = MTLSizeMake(blockSize, blockSize, 1);
//    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);
    const MTLSize threads = MTLSizeMake(roi.width(), roi.height(), 1);

    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);

    NSArray* args = @[
                [out_upscaledMap_dmp getBuffer],
                @([out_upscaledMap_dmp getBytesUpToDim:0]),
                [in_map_dmp getBuffer],
                @([in_map_dmp getBytesUpToDim:0]),
                @(ratio),
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
    ];

    // kernel execution
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::mapUpscale_kernel" Args:args];
    
//    mapUpscale_kernel<float3><<<grid, block, 0, stream>>>(
//        out_upscaledMap_dmp.getBuffer(),
//        out_upscaledMap_dmp.getPitch(),
//        in_map_dmp.getBuffer(),
//        in_map_dmp.getPitch(),
//        ratio,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void depthThicknessSmoothThickness(DeviceBuffer* inout_depthThicknessMap_dmp,
                                               const SgmParams& sgmParams,
                                               const RefineParams& refineParams,
                                               const ROI& roi)
{
    const int sgmScaleStep = sgmParams.scale * sgmParams.stepXY;
    const int refineScaleStep = refineParams.scale * refineParams.stepXY;

    // min/max number of Refine samples in SGM thickness area
    const float minNbRefineSamples = 2.f;
    const float maxNbRefineSamples = fmax(sgmScaleStep / float(refineScaleStep), minNbRefineSamples);

    // min/max SGM thickness inflate factor
    const float minThicknessInflate = refineParams.halfNbDepths / maxNbRefineSamples;
    const float maxThicknessInflate = refineParams.halfNbDepths / minNbRefineSamples;

    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = 8;
//    MTLSize gridSize = MTLSizeMake(divUp(roi.width(), threadGroupSize), divUp(roi.height(), threadGroupSize), 1);
    MTLSize threadsSize = MTLSizeMake(roi.width(), roi.height(), 1);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, threadGroupSize, 1);
    
    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    
    NSArray* args = @[
        [inout_depthThicknessMap_dmp getBuffer],
        @([inout_depthThicknessMap_dmp getBytesUpToDim:0]),
        @(minThicknessInflate),
        @(maxThicknessInflate),
        [NSData dataWithBytes:&roi_d length:sizeof(ROI_d)]
    ];
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threadsSize ThreadgroupSize:threadgroupSize KernelFuncName:@"depthMap::depthThicknessMapSmoothThickness_kernel" Args:args];
    
    
    // kernel launch parameters
//    const int blockSize = 8;
//    const dim3 block(blockSize, blockSize, 1);
//    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);
//
//    // kernel execution
//    depthThicknessMapSmoothThickness_kernel<<<grid, block, 0, stream>>>(
//        inout_depthThicknessMap_dmp.getBuffer(),
//        inout_depthThicknessMap_dmp.getPitch(),
//        minThicknessInflate,
//        maxThicknessInflate,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void computeSgmUpscaledDepthPixSizeMap(DeviceBuffer* out_upscaledDepthPixSizeMap_dmp,
                                                     DeviceBuffer* in_sgmDepthThicknessMap_dmp,
                                                     DeviceCameraParams const& rcDeviceCameraParams,
                                                     const DeviceMipmapImage& rcDeviceMipmapImage,
                                                     const RefineParams& refineParams,
                                                     const ROI& roi)
{
    // compute upscale ratio
    MTLSize out_mapDim = [out_upscaledDepthPixSizeMap_dmp getSize];
    MTLSize in_mapDim = [in_sgmDepthThicknessMap_dmp getSize];
    const float ratio = float(in_mapDim.width)/float(out_mapDim.width);
//    const CudaSize<2>& out_mapDim = out_upscaledDepthPixSizeMap_dmp.getSize();
//    const CudaSize<2>& in_mapDim = in_sgmDepthThicknessMap_dmp.getSize();
//    const float ratio = float(in_mapDim.x()) / float(out_mapDim.x());
//
//    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const MTLSize& rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
//    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
//
    // kernel launch parameters
    const int blockSize = 16;
    const MTLSize block = MTLSizeMake(blockSize, blockSize, 1);
    const MTLSize threads = MTLSizeMake(roi.width(), roi.height(), 1);
//    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);

    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);

    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    // kernel execution
    if(refineParams.interpolateMiddleDepth)
    {

        NSArray* args = @[
                        [out_upscaledDepthPixSizeMap_dmp getBuffer],
                        @([out_upscaledDepthPixSizeMap_dmp getBytesUpToDim:0]),
                        [in_sgmDepthThicknessMap_dmp getBuffer],
                        @([in_sgmDepthThicknessMap_dmp getBytesUpToDim:0]),
                        [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                        rcDeviceMipmapImage.getTextureObject(),
                        @((unsigned int)(rcLevelDim.width)),
                        @((unsigned int)(rcLevelDim.height)),
                        @(rcMipmapLevel),
                        @(refineParams.stepXY),
                        @(refineParams.halfNbDepths),
                        @(ratio),
                        [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
        ];

        [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::computeSgmUpscaledDepthPixSizeMap_bilinear_kernel" Args:args];

//        computeSgmUpscaledDepthPixSizeMap_bilinear_kernel<<<grid, block, 0, stream>>>(
//            out_upscaledDepthPixSizeMap_dmp.getBuffer(),
//            out_upscaledDepthPixSizeMap_dmp.getPitch(),
//            in_sgmDepthThicknessMap_dmp.getBuffer(),
//            in_sgmDepthThicknessMap_dmp.getPitch(),
//            rcDeviceCameraParamsId,
//            rcDeviceMipmapImage.getTextureObject(),
//            (unsigned int)(rcLevelDim.x()),
//            (unsigned int)(rcLevelDim.y()),
//            rcMipmapLevel,
//            refineParams.stepXY,
//            refineParams.halfNbDepths,
//            ratio,
//            roi);
    }
    else
    {
        NSArray* args = @[
                        [out_upscaledDepthPixSizeMap_dmp getBuffer],
                        @([out_upscaledDepthPixSizeMap_dmp getBytesUpToDim:0]),
                        [in_sgmDepthThicknessMap_dmp getBuffer],
                        @([in_sgmDepthThicknessMap_dmp getBytesUpToDim:0]),
                        [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                        rcDeviceMipmapImage.getTextureObject(),
                        @((unsigned int)(rcLevelDim.width)),
                        @((unsigned int)(rcLevelDim.height)),
                        @(rcMipmapLevel),
                        @(refineParams.stepXY),
                        @(refineParams.halfNbDepths),
                        @(ratio),
                        [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
        ];

        [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::computeSgmUpscaledDepthPixSizeMap_nearestNeighbor_kernel" Args:args];
        
        DeviceTexture* texture_in = [in_sgmDepthThicknessMap_dmp getDebugTexture];
        DeviceTexture* texture = [out_upscaledDepthPixSizeMap_dmp getDebugTexture];
        NSLog(@"xxx");
//        computeSgmUpscaledDepthPixSizeMap_nearestNeighbor_kernel<<<grid, block, 0, stream>>>(
//            out_upscaledDepthPixSizeMap_dmp.getBuffer(),
//            out_upscaledDepthPixSizeMap_dmp.getPitch(),
//            in_sgmDepthThicknessMap_dmp.getBuffer(),
//            in_sgmDepthThicknessMap_dmp.getPitch(),
//            rcDeviceCameraParamsId,
//            rcDeviceMipmapImage.getTextureObject(),
//            (unsigned int)(rcLevelDim.x()),
//            (unsigned int)(rcLevelDim.y()),
//            rcMipmapLevel,
//            refineParams.stepXY,
//            refineParams.halfNbDepths,
//            ratio,
//            roi);
    }
}

void depthSimMapComputeNormal(DeviceBuffer* out_normalMap_dmp,
                                            DeviceBuffer* in_depthSimMap_dmp,
                              DeviceCameraParams const& rcDeviceCameraParams,
                                            const int stepXY,
                                            const ROI& roi)
{
    // kernel launch parameters
    NSUInteger threadGroupSize = 8;
//    MTLSize gridSize = MTLSizeMake(divUp(roi.width(), threadGroupSize), divUp(roi.height(), threadGroupSize), 1);
    MTLSize threads = MTLSizeMake(roi.width(), roi.height(), 1);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, threadGroupSize, 1);

    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    
    NSArray* args = @[
                [out_normalMap_dmp getBuffer],
                @([out_normalMap_dmp getBytesUpToDim:0]),
                [in_depthSimMap_dmp getBuffer],
                @([in_depthSimMap_dmp getBytesUpToDim:0]),
                [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                @(stepXY),
                @(3),/* wsh */
                [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
    ];

    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    [pipeline Exec:threads ThreadgroupSize:threadgroupSize KernelFuncName:@"depthMap::depthSimMapComputeNormal_kernel" Args:args];
    
    //TODO: check normal...
    
//    const dim3 block(8, 8, 1);
//    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);

    // kernel execution
//    depthSimMapComputeNormal_kernel<3 /* wsh */><<<grid, block, 0, stream>>>(
//        out_normalMap_dmp.getBuffer(),
//        out_normalMap_dmp.getPitch(),
//        in_depthSimMap_dmp.getBuffer(),
//        in_depthSimMap_dmp.getPitch(),
//        rcDeviceCameraParamsId,
//        stepXY,
//        roi);
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

void depthSimMapOptimizeGradientDescent(DeviceBuffer* out_optimizeDepthSimMap_dmp,
                                                      DeviceBuffer* inout_imgVariance_dmp,
                                                      DeviceBuffer* inout_tmpOptDepthMap_dmp,
                                                      DeviceBuffer* in_sgmDepthPixSizeMap_dmp,
                                                      DeviceBuffer* in_refineDepthSimMap_dmp,
                                                      DeviceCameraParams const& rcDeviceCameraParams,
                                                      const DeviceMipmapImage& rcDeviceMipmapImage,
                                                      const RefineParams& refineParams,
                                                      const ROI& roi)
{
    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const MTLSize& rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);

    // initialize depth/sim map optimized with SGM depth/pixSize map
    [out_optimizeDepthSimMap_dmp copyFrom:in_sgmDepthPixSizeMap_dmp];
//    out_optimizeDepthSimMap_dmp.copyFrom(in_sgmDepthPixSizeMap_dmp, stream);

    ROI_d roi_d;
    roi_d.lt = simd_make_float2(roi.x.begin, roi.y.begin);
    roi_d.rb = simd_make_float2(roi.x.end, roi.y.end);
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    
    {
        // kernel launch parameters
        const MTLSize& lblock = MTLSizeMake(32, 2, 1);
        const MTLSize& lthreads = MTLSizeMake(roi.width(), roi.height(), 1);
//        const dim3 lgrid(divUp(roi.width(), lblock.x), divUp(roi.height(), lblock.y), 1);

        // kernel execution
        
        
        NSArray* args = @[
                        [inout_imgVariance_dmp getBuffer],
                        @([inout_imgVariance_dmp getBytesUpToDim:0]),
                        rcDeviceMipmapImage.getTextureObject(),
                        @((unsigned int)(rcLevelDim.width)),
                        @((unsigned int)(rcLevelDim.height)),
                        @(rcMipmapLevel),
                        @(refineParams.stepXY),
                        [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
        ];

        
        [pipeline Exec:lthreads ThreadgroupSize:lblock KernelFuncName:@"depthMap::optimize_varLofLABtoW_kernel" Args:args];
//        optimize_varLofLABtoW_kernel<<<lgrid, lblock, 0, stream>>>(
//            inout_imgVariance_dmp.getBuffer(), 
//            inout_imgVariance_dmp.getPitch(),
//            rcDeviceMipmapImage.getTextureObject(),
//            (unsigned int)(rcLevelDim.x()),
//            (unsigned int)(rcLevelDim.y()),
//            rcMipmapLevel,
//            refineParams.stepXY,
//            roi);
    }

    id<MTLTexture> imgVarianceTex = [DeviceTexture initWithFloatBuffer:inout_imgVariance_dmp];
    id<MTLTexture> depthTex = [DeviceTexture initWithFloatBuffer:inout_tmpOptDepthMap_dmp];
//    CudaTexture<float, false, false> imgVarianceTex(inout_imgVariance_dmp); // neighbor interpolation, without normalized coordinates
//    CudaTexture<float, false, false> depthTex(inout_tmpOptDepthMap_dmp);    // neighbor interpolation, without normalized coordinates

    // kernel launch parameters
    const int blockSize = 16;
    const MTLSize& block = MTLSizeMake(blockSize, blockSize, 1);
//    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);
    const MTLSize& threads = MTLSizeMake(roi.width(), roi.height(), 1);

    for(int iter = 0; iter < refineParams.optimizationNbIterations; ++iter) // default nb iterations is 100
    {
        // copy depths values from out_depthSimMapOptimized_dmp to inout_tmpOptDepthMap_dmp
        {
            NSArray* args = @[
                            [inout_tmpOptDepthMap_dmp getBuffer],
                            @([inout_tmpOptDepthMap_dmp getBytesUpToDim:0]),
                            [out_optimizeDepthSimMap_dmp getBuffer], // initialized with SGM depth/pixSize map
                            @([out_optimizeDepthSimMap_dmp getBytesUpToDim:0]),
                            [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
            ];
            [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::optimize_getOptDeptMapFromOptDepthSimMap_kernel" Args:args];
        }
//        optimize_getOptDeptMapFromOptDepthSimMap_kernel<<<grid, block, 0, stream>>>(
//            inout_tmpOptDepthMap_dmp.getBuffer(), 
//            inout_tmpOptDepthMap_dmp.getPitch(), 
//            out_optimizeDepthSimMap_dmp.getBuffer(), // initialized with SGM depth/pixSize map
//            out_optimizeDepthSimMap_dmp.getPitch(),
//            roi);

        // adjust depth/sim by using previously computed depths
        {
            NSArray* args = @[
                            [out_optimizeDepthSimMap_dmp getBuffer],
                            @([out_optimizeDepthSimMap_dmp getBytesUpToDim:0]),
                            [in_sgmDepthPixSizeMap_dmp getBuffer],
                            @([in_sgmDepthPixSizeMap_dmp getBytesUpToDim:0]),
                            [in_refineDepthSimMap_dmp getBuffer],
                            @([in_refineDepthSimMap_dmp getBytesUpToDim:0]),
                            [NSData dataWithBytes:&rcDeviceCameraParams length:sizeof(rcDeviceCameraParams)],
                            imgVarianceTex,
                            depthTex,
                            @(iter),
                            [NSData dataWithBytes:&roi_d length:sizeof(roi_d)]
            ];
            [pipeline Exec:threads ThreadgroupSize:block KernelFuncName:@"depthMap::optimize_depthSimMap_kernel" Args:args];
        }
//        optimize_depthSimMap_kernel<<<grid, block, 0, stream>>>(
//            out_optimizeDepthSimMap_dmp.getBuffer(),
//            out_optimizeDepthSimMap_dmp.getPitch(),
//            in_sgmDepthPixSizeMap_dmp.getBuffer(),
//            in_sgmDepthPixSizeMap_dmp.getPitch(),
//            in_refineDepthSimMap_dmp.getBuffer(),
//            in_refineDepthSimMap_dmp.getPitch(),
//            rcDeviceCameraParamsId,
//            imgVarianceTex.textureObj,
//            depthTex.textureObj,
//            iter, 
//            roi);
    }
//
//    // check cuda last error
//    CHECK_CUDA_ERROR();
}

} // namespace depthMap

