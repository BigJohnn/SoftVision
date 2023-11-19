// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Refine.hpp"

#include <softvision_omp.hpp>
#include <SoftVisionLog.h>
#include <mvsData/Point2d.hpp>
#include <mvsData/Point3d.hpp>
#include <mvsUtils/fileIO.hpp>
#include <depthMap/depthMapUtils.hpp>
#include <depthMap/volumeIO.hpp>
#include <depthMap/gpu/host/DeviceCache.hpp>
#include <depthMap/gpu/planeSweeping/deviceDepthSimilarityMap.hpp>
#include <depthMap/gpu/planeSweeping/deviceSimilarityVolume.hpp>


namespace depthMap {

Refine::Refine(const mvsUtils::MultiViewParams& mp,
               const mvsUtils::TileParams& tileParams, 
               const RefineParams& refineParams)
    : _mp(mp)
    , _tileParams(tileParams)
    , _refineParams(refineParams)
    
{
    // get tile maximum dimensions
    const int downscale = _refineParams.scale * _refineParams.stepXY;
    const int maxTileWidth  = divideRoundUp(tileParams.bufferWidth , downscale);
    const int maxTileHeight = divideRoundUp(tileParams.bufferHeight, downscale);

    // compute depth/sim map maximum dimensions
//    const CudaSize<2> depthSimMapDim(maxTileWidth, maxTileHeight);
    MTLSize depthSimMapDim = MTLSizeMake(maxTileWidth, maxTileHeight, 1);

    // allocate depth/sim maps in device memory
    [_sgmDepthPixSizeMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(simd_float2)];
    [_refinedDepthSimMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(simd_float2)];
    [_optimizedDepthSimMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(simd_float2)];
//    _sgmDepthPixSizeMap_dmp.allocate(depthSimMapDim);
//    _refinedDepthSimMap_dmp.allocate(depthSimMapDim);
//    _optimizedDepthSimMap_dmp.allocate(depthSimMapDim);

    // allocate SGM upscaled normal map in device memory
    if(_refineParams.useSgmNormalMap)
        [_sgmNormalMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(simd_float3)];
//        _sgmNormalMap_dmp.allocate(depthSimMapDim);

    // allocate normal map in device memory
    if(_refineParams.exportIntermediateNormalMaps)
        [_normalMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(simd_float3)];
//        _normalMap_dmp.allocate(depthSimMapDim);

    // compute volume maximum dimensions
    const int nbDepthsToRefine = _refineParams.halfNbDepths * 2 + 1;
//    const CudaSize<3> volDim(maxTileWidth, maxTileHeight, nbDepthsToRefine);
    MTLSize volDim = MTLSizeMake(maxTileWidth, maxTileHeight, nbDepthsToRefine);

    // allocate refine volume in device memory
    [_volumeRefineSim_dmp allocate:volDim elemSizeInBytes:sizeof(TSimRefine)];
//    _volumeRefineSim_dmp.allocate(volDim);

    // allocate depth/sim map optimization buffers
    if(_refineParams.useColorOptimization)
    {
        [_optTmpDepthMap_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(float)];
        [_optImgVariance_dmp allocate:depthSimMapDim elemSizeInBytes:sizeof(float)];
//        _optTmpDepthMap_dmp.allocate(depthSimMapDim);
//        _optImgVariance_dmp.allocate(depthSimMapDim);
    }
}

double Refine::getDeviceMemoryConsumption() const
{
    size_t bytes = 0;

    bytes += [_sgmDepthPixSizeMap_dmp getBufferLength];
    bytes += [_refinedDepthSimMap_dmp getBufferLength];
    bytes += [_optimizedDepthSimMap_dmp getBufferLength];
    bytes += [_sgmNormalMap_dmp getBufferLength];
    bytes += [_normalMap_dmp getBufferLength];
    bytes += [_volumeRefineSim_dmp getBufferLength];
    bytes += [_optTmpDepthMap_dmp getBufferLength];
    bytes += [_optImgVariance_dmp getBufferLength];

    return (double(bytes) / (1024.0 * 1024.0));
}

//double Refine::getDeviceMemoryConsumptionUnpadded() const
//{
//    size_t bytes = 0;
//
//    bytes += _sgmDepthPixSizeMap_dmp.getBytesUnpadded();
//    bytes += _refinedDepthSimMap_dmp.getBytesUnpadded();
//    bytes += _optimizedDepthSimMap_dmp.getBytesUnpadded();
//    bytes += _sgmNormalMap_dmp.getBytesUnpadded();
//    bytes += _normalMap_dmp.getBytesUnpadded();
//    bytes += _volumeRefineSim_dmp.getBytesUnpadded();
//    bytes += _optTmpDepthMap_dmp.getBytesUnpadded();
//    bytes += _optImgVariance_dmp.getBytesUnpadded();
//
//    return (double(bytes) / (1024.0 * 1024.0));
//}

void Refine::refineRc(const Tile& tile, DeviceBuffer* in_sgmDepthThicknessMap_dmp, DeviceBuffer* in_sgmNormalMap_dmp)
{
    const IndexT viewId = _mp.getViewId(tile.rc);

    LOG_X(tile << "Refine depth/sim map of view id: " << viewId << ", rc: " << tile.rc << " (" << (tile.rc + 1) << " / " << _mp.ncams << ").");

    // compute upscaled SGM depth/pixSize map
    // compute upscaled SGM normal map
    {
        // downscale the region of interest
        const ROI downscaledRoi = downscaleROI(tile.roi, _refineParams.scale * _refineParams.stepXY);

        // get device cache instance
        DeviceCache& deviceCache = DeviceCache::getInstance();

        // get R device camera parameters id from cache
        const int rcDeviceCameraParamsId = deviceCache.requestCameraParamsId(tile.rc, _refineParams.scale, _mp);

        // get R device mipmap image from cache
        const DeviceMipmapImage& rcDeviceMipmapImage = deviceCache.requestMipmapImage(tile.rc, _mp);

        // compute upscaled SGM depth/pixSize map
        // - upscale SGM depth/thickness map
        // - filter masked pixels (alpha)
        // - compute pixSize from SGM thickness
        computeSgmUpscaledDepthPixSizeMap(_sgmDepthPixSizeMap_dmp,
                                               in_sgmDepthThicknessMap_dmp,
                                               rcDeviceCameraParamsId,
                                               rcDeviceMipmapImage,
                                               _refineParams,
                                               downscaledRoi);

        // export intermediate depth/pixSize map (if requested by user)
        if(_refineParams.exportIntermediateDepthSimMaps)
            writeDepthPixSizeMap(tile.rc, _mp, _tileParams, tile.roi, _sgmDepthPixSizeMap_dmp, _refineParams.scale, _refineParams.stepXY, "sgmUpscaled");

        // upscale SGM normal map (if needed)
        if(_refineParams.useSgmNormalMap && [in_sgmNormalMap_dmp getBufferPtr] != nullptr)
        {
            cuda_normalMapUpscale(_sgmNormalMap_dmp, in_sgmNormalMap_dmp, downscaledRoi);
        }
    }

    // refine and fuse depth/sim map
    if(_refineParams.useRefineFuse)
    {
        // refine and fuse with volume strategy
        refineAndFuseDepthSimMap(tile);
    }
    else
    {
        LOG_X(tile << "Refine and fuse depth/sim map volume disabled.");
        depthSimMapCopyDepthOnly(_refinedDepthSimMap_dmp, _sgmDepthPixSizeMap_dmp, 1.0f);
    }

    // export intermediate depth/sim map (if requested by user)
    if(_refineParams.exportIntermediateDepthSimMaps)
      writeDepthSimMap(tile.rc, _mp, _tileParams, tile.roi, _refinedDepthSimMap_dmp, _refineParams.scale, _refineParams.stepXY, "refinedFused");

    // export intermediate normal map (if requested by user)
    if(_refineParams.exportIntermediateNormalMaps)
      computeAndWriteNormalMap(tile, _refinedDepthSimMap_dmp, "refinedFused");

    // optimize depth/sim map
    if(_refineParams.useColorOptimization && _refineParams.optimizationNbIterations > 0)
    {
        optimizeDepthSimMap(tile);
    }
    else
    {
        LOG_X(tile << "Color optimize depth/sim map disabled. //TODO: check");
//        _optimizedDepthSimMap_dmp.copyFrom(_refinedDepthSimMap_dmp);
    }

    // export intermediate normal map (if requested by user)
    if(_refineParams.exportIntermediateNormalMaps)
      computeAndWriteNormalMap(tile, _optimizedDepthSimMap_dmp);

    LOG_X(tile << "Refine depth/sim map done.");
}

void Refine::refineAndFuseDepthSimMap(const Tile& tile)
{
    LOG_X(tile << "Refine and fuse depth/sim map volume.");

    // downscale the region of interest
    const ROI downscaledRoi = downscaleROI(tile.roi, _refineParams.scale * _refineParams.stepXY);

    // get the depth range
    const Range depthRange(0, [_volumeRefineSim_dmp getSize].depth);

    // initialize the similarity volume at 0
    // each tc filtered and inverted similarity value will be summed in this volume
    volumeInitialize(_volumeRefineSim_dmp, TSimRefine(0.f));

    // get device cache instance
    DeviceCache& deviceCache = DeviceCache::getInstance();

    // get R device camera parameters id from cache
    const int rcDeviceCameraParamsId = deviceCache.requestCameraParamsId(tile.rc, _refineParams.scale, _mp);
    id<MTLBuffer> rcDeviceCameraParams = deviceCache.requestCameraParamsBuffer(tile.rc, _refineParams.scale, _mp);
    
    // get R device mipmap image from cache
    const DeviceMipmapImage& rcDeviceMipmapImage = deviceCache.requestMipmapImage(tile.rc, _mp);

    // compute for each RcTc each similarity value for each depth to refine
    // sum the inverted / filtered similarity value, best value is the HIGHEST
    for(std::size_t tci = 0; tci < tile.refineTCams.size(); ++tci)
    {
        const int tc = tile.refineTCams.at(tci);

        // get T device camera parameters id from cache
        const int tcDeviceCameraParamsId = deviceCache.requestCameraParamsId(tc, _refineParams.scale, _mp);
        id<MTLBuffer> tcDeviceCameraParams = deviceCache.requestCameraParamsBuffer(tc, _refineParams.scale, _mp);

        // get T device mipmap image from cache
        const DeviceMipmapImage& tcDeviceMipmapImage = deviceCache.requestMipmapImage(tc, _mp);

        LOG_X(tile << "Refine similarity volume:" << std::endl
                                   << "\t- rc: " << tile.rc << std::endl
                                   << "\t- tc: " << tc << " (" << (tci + 1) << "/" << tile.refineTCams.size() << ")" << std::endl
                                   << "\t- rc camera parameters id: " << rcDeviceCameraParamsId << std::endl
                                   << "\t- tc camera parameters id: " << tcDeviceCameraParamsId << std::endl
                                   << "\t- tile range x: [" << downscaledRoi.x.begin << " - " << downscaledRoi.x.end << "]" << std::endl
                                   << "\t- tile range y: [" << downscaledRoi.y.begin << " - " << downscaledRoi.y.end << "]" << std::endl);

        volumeRefineSimilarity(_volumeRefineSim_dmp, 
                                    _sgmDepthPixSizeMap_dmp,
                                    (_refineParams.useSgmNormalMap) ? _sgmNormalMap_dmp : nil,
                                    rcDeviceCameraParams,
                                    tcDeviceCameraParams,
                                    rcDeviceMipmapImage,
                                    tcDeviceMipmapImage,
                                    _refineParams, 
                                    depthRange,
                                    downscaledRoi);
    }

    // export intermediate volume information (if requested by user)
    exportVolumeInformation(tile, "afterRefine");

    // retrieve the best depth/sim in the volume
    // compute sub-pixel sample using a sliding gaussian 
    cuda_volumeRefineBestDepth(_refinedDepthSimMap_dmp, 
                               _sgmDepthPixSizeMap_dmp,
                               _volumeRefineSim_dmp,
                               _refineParams,
                               downscaledRoi);
    
    LOG_X(tile << "Refine and fuse depth/sim map volume done.");
}

void Refine::optimizeDepthSimMap(const Tile& tile)
{
    LOG_X(tile << "Color optimize depth/sim map.");

    // downscale the region of interest
    const ROI downscaledRoi = downscaleROI(tile.roi, _refineParams.scale * _refineParams.stepXY);
    
    // get R device camera from cache
    DeviceCache& deviceCache = DeviceCache::getInstance();

    // get R device camera parameters id from cache
    const int rcDeviceCameraParamsId = deviceCache.requestCameraParamsId(tile.rc, _refineParams.scale, _mp);

    // get R device mipmap image from cache
    const DeviceMipmapImage& rcDeviceMipmapImage = deviceCache.requestMipmapImage(tile.rc, _mp);

    cuda_depthSimMapOptimizeGradientDescent(_optimizedDepthSimMap_dmp, // output depth/sim map optimized
                                            _optImgVariance_dmp,       // image variance buffer pre-allocate
                                            _optTmpDepthMap_dmp,       // temporary depth map buffer pre-allocate
                                            _sgmDepthPixSizeMap_dmp,   // input SGM upscaled depth/pixSize map
                                            _refinedDepthSimMap_dmp,   // input refined and fused depth/sim map
                                            rcDeviceCameraParamsId,
                                            rcDeviceMipmapImage,
                                            _refineParams,
                                            downscaledRoi);

    LOG_X(tile << "Color optimize depth/sim map done.");
}

void Refine::computeAndWriteNormalMap(const Tile& tile, DeviceBuffer* in_depthSimMap_dmp, const std::string& name)
{
    // downscale the region of interest
    const ROI downscaledRoi = downscaleROI(tile.roi, _refineParams.scale * _refineParams.stepXY);

    // get R device camera parameters id from cache
    DeviceCache& deviceCache = DeviceCache::getInstance();
    const int rcDeviceCameraParamsId = deviceCache.requestCameraParamsId(tile.rc, _refineParams.scale, _mp);

    LOG_X(tile << "Refine compute normal map of view id: " << _mp.getViewId(tile.rc) << ", rc: " << tile.rc << " (" << (tile.rc + 1) << " / " << _mp.ncams << ").");

    cuda_depthSimMapComputeNormal(_normalMap_dmp, in_depthSimMap_dmp, rcDeviceCameraParamsId, _refineParams.stepXY, downscaledRoi);

    writeNormalMap(tile.rc, _mp, _tileParams, tile.roi, _normalMap_dmp, _refineParams.scale, _refineParams.stepXY, name);
}

void Refine::exportVolumeInformation(const Tile& tile, const std::string& name) const
{
    if(!_refineParams.exportIntermediateCrossVolumes &&
       !_refineParams.exportIntermediateVolume9pCsv)
    {
        // nothing to do
        return;
    }

    // get tile begin indexes (default no tile)
    int tileBeginX = -1;
    int tileBeginY = -1;

    if(tile.nbTiles > 1)
    {
        tileBeginX = tile.roi.x.begin;
        tileBeginY = tile.roi.y.begin;
    }

    // copy device similarity volume to host memory
//    CudaHostMemoryHeap<TSimRefine, 3> volumeSim_hmh(_volumeRefineSim_dmp.getSize());
//    volumeSim_hmh.copyFrom(_volumeRefineSim_dmp);

    // copy device SGM upscale depth/sim map to host memory
//    CudaHostMemoryHeap<float2, 2> depthPixSizeMapSgmUpscale_hmh(_sgmDepthPixSizeMap_dmp.getSize());
//    depthPixSizeMapSgmUpscale_hmh.copyFrom(_sgmDepthPixSizeMap_dmp);

    if(_refineParams.exportIntermediateCrossVolumes)
    {
        LOG_X(tile << "Export similarity volume cross (" << name << ").");

        const std::string volumeCrossPath = getFileNameFromIndex(_mp, tile.rc, mvsUtils::EFileType::volumeCross, "_" + name, tileBeginX, tileBeginY);

//        exportSimilarityVolumeCross(volumeSim_hmh, depthPixSizeMapSgmUpscale_hmh, _mp, tile.rc, _refineParams, volumeCrossPath, tile.roi);

        LOG_X(tile << "Export similarity volume cross (" << name << ") done.");
    }

    if(_refineParams.exportIntermediateTopographicCutVolumes)
    {
        LOG_X(tile << "Export similarity volume topographic cut (" << name << ").");

        const std::string volumeCutPath = getFileNameFromIndex(_mp, tile.rc, mvsUtils::EFileType::volumeTopographicCut, "_" + name, tileBeginX, tileBeginY);

//        exportSimilarityVolumeTopographicCut(volumeSim_hmh, depthPixSizeMapSgmUpscale_hmh, _mp, tile.rc, _refineParams, volumeCutPath, tile.roi);

        LOG_X(tile << "Export similarity volume topographic cut (" << name << ") done.");
    }

    if(_refineParams.exportIntermediateVolume9pCsv)
    {
        LOG_X(tile << "Export similarity volume 9 points CSV (" << name << ").");

        const std::string stats9Path = getFileNameFromIndex(_mp, tile.rc, mvsUtils::EFileType::stats9p, "_refine", tileBeginX, tileBeginY);

//        exportSimilaritySamplesCSV(volumeSim_hmh, name, _refineParams, stats9Path, tile.roi);

        LOG_X(tile << "Export similarity volume 9 points CSV (" << name << ") done.");
    }
}

} // namespace depthMap

