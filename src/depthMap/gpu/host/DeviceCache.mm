// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
#import <Metal/Metal.h>
#import <depthMap/gpu/host/memory.hpp>
#import <depthMap/gpu/host/DeviceTexture.hpp>

#include "DeviceCache.hpp"

#include <SoftVisionLog.h>
#include <depthMap/gpu/host/utils.hpp>
#include <depthMap/gpu/device/DeviceCameraParams.hpp>
//#include <depthMap/gpu/imageProcessing/deviceGaussianFilter.hpp>


#include <simd/simd.h>

// maximum pre-computed Gaussian scales
#define DEVICE_MAX_DOWNSCALE  ( MAX_CONSTANT_GAUSS_SCALES - 1 )

namespace depthMap {

void normalize(vector_float3& a)
{
    float d = sqrt(a.x * a.x + a.y * a.y + a.z * a.z); //TODO: check whether should be vector_float
    a.x /= d;
    a.y /= d;
    a.z /= d;
}

void fillCameraParameters(DeviceCameraParams &cameraParameters_h, int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    Matrix3x3 scaleM;
    scaleM.m11 = 1.0 / float(downscale);
    scaleM.m12 = 0.0;
    scaleM.m13 = 0.0;
    scaleM.m21 = 0.0;
    scaleM.m22 = 1.0 / float(downscale);
    scaleM.m23 = 0.0;
    scaleM.m31 = 0.0;
    scaleM.m32 = 0.0;
    scaleM.m33 = 1.0;

    Matrix3x3 K = scaleM * mp.KArr[camId];
    Matrix3x3 iK = K.inverse();
    Matrix3x4 P = K * (mp.RArr[camId] | (Point3d(0.0, 0.0, 0.0) - mp.RArr[camId] * mp.CArr[camId]));
    Matrix3x3 iP = mp.iRArr[camId] * iK;

    cameraParameters_h.C.x = mp.CArr[camId].x;
    cameraParameters_h.C.y = mp.CArr[camId].y;
    cameraParameters_h.C.z = mp.CArr[camId].z;

    cameraParameters_h.P = simd_matrix(
                                         simd_make_float3(P.m11, P.m21, P.m31),
                                         simd_make_float3(P.m12, P.m22, P.m32),
                                         simd_make_float3(P.m13, P.m23, P.m33),
                                         simd_make_float3(P.m14, P.m24, P.m34));
    
    cameraParameters_h.iP = simd_matrix(
                                         simd_make_float3(iP.m11, iP.m21, iP.m31),
                                         simd_make_float3(iP.m12, iP.m22, iP.m32),
                                         simd_make_float3(iP.m13, iP.m23, iP.m33));
    
    auto&& R = mp.RArr[camId];
    cameraParameters_h.R = simd_matrix(
                                         simd_make_float3(R.m11, R.m21, R.m31),
                                         simd_make_float3(R.m12, R.m22, R.m32),
                                         simd_make_float3(R.m13, R.m23, R.m33));

    
    auto&& iR = mp.iRArr[camId];
    cameraParameters_h.iR = simd_matrix(
                                         simd_make_float3(iR.m11, iR.m21, iR.m31),
                                         simd_make_float3(iR.m12, iR.m22, iR.m32),
                                         simd_make_float3(iR.m13, iR.m23, iR.m33));
    
    cameraParameters_h.K = simd_matrix(
                                         simd_make_float3(K.m11, K.m21, K.m31),
                                         simd_make_float3(K.m12, K.m22, K.m32),
                                         simd_make_float3(K.m13, K.m23, K.m33));
    
    cameraParameters_h.iK = simd_matrix(
                                         simd_make_float3(iK.m11, iK.m21, iK.m31),
                                         simd_make_float3(iK.m12, iK.m22, iK.m32),
                                         simd_make_float3(iK.m13, iK.m23, iK.m33));

    cameraParameters_h.XVect = simd_mul(cameraParameters_h.iR, simd_make_float3(1.f, 0.f, 0.f));
    normalize(cameraParameters_h.XVect);

    cameraParameters_h.YVect = simd_mul(cameraParameters_h.iR, simd_make_float3(0.f, 1.f, 0.f));
    normalize(cameraParameters_h.YVect);

    cameraParameters_h.ZVect = simd_mul(cameraParameters_h.iR, simd_make_float3(0.f, 0.f, 1.f));
    normalize(cameraParameters_h.ZVect);
    
    
}

DeviceCache::SingleDeviceCache::SingleDeviceCache(int maxMipmapImages, int maxCameraParams)
    : mipmapCache(maxMipmapImages)
    , cameraParamCache(maxCameraParams)
{
    // get the current device id
    const int cudaDeviceId = getGpuDeviceId();

    LOG_X("Initialize device cache (device id: " << cudaDeviceId << "):" << std::endl
                          << "\t - # mipmap images: " << maxMipmapImages << std::endl
                          << "\t - # cameras parameters: " << maxCameraParams);

    // initialize Gaussian filters in GPU constant memory
    // force at compilation to build with maximum pre-computed Gaussian scales
    // note: useful for downscale with gaussian blur, volume gaussian blur (Z, XYZ)
//    createConstantGaussianArray(cudaDeviceId, DEVICE_MAX_DOWNSCALE); //TODO: use MPS?

    // the maximum number of camera parameters in device cache cannot be superior
    // to the number of camera parameters in the array in device constant memory
    if(maxCameraParams > ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS)
        ALICEVISION_THROW_ERROR("Cannot initialize device cache with more than " << ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS << " camera parameters (device id: " << cudaDeviceId << ", # cameras parameters: " << maxCameraParams << ").")

    // initialize cached mipmap image containers
    mipmaps.reserve(maxMipmapImages);
    for(int i = 0; i < maxMipmapImages; ++i)
    {
        mipmaps.push_back(std::make_unique<DeviceMipmapImage>());
    }
}

void DeviceCache::clear()
{
    // get the current device id
    const int cudaDeviceId = getGpuDeviceId();

    // find the current SingleDeviceCache
    auto it = _cachePerDevice.find(cudaDeviceId);

    // if found, erase SingleDeviceCache data
    if(it != _cachePerDevice.end())
        _cachePerDevice.erase(it);
}

void DeviceCache::build(int maxMipmapImages, int maxCameraParams)
{
    // get the current device id
    const int cudaDeviceId = getGpuDeviceId();

    // reset the current device cache
    _cachePerDevice[cudaDeviceId].reset(new SingleDeviceCache(maxMipmapImages, maxCameraParams));
}

DeviceCache::SingleDeviceCache& DeviceCache::getCurrentDeviceCache()
{
    // get the current device id
    const int cudaDeviceId = getGpuDeviceId();

    // find the current SingleDeviceCache
    auto it = _cachePerDevice.find(cudaDeviceId);

    // check found and initialized
    if(it == _cachePerDevice.end() || it->second == nullptr)
    {
        ALICEVISION_THROW_ERROR("Device cache is not initialized (cuda device id: " << cudaDeviceId <<").")
    }

    // return current SingleDeviceCache reference
    return *(it->second);
}

void DeviceCache::addMipmapImage(int camId,
                                 int minDownscale,
                                 int maxDownscale,
                                 mvsUtils::ImagesCache<image::Image<image::RGBAfColor>>& imageCache,
                                 const mvsUtils::MultiViewParams& mp)
{
    // get current device cache
    SingleDeviceCache& currentDeviceCache = getCurrentDeviceCache();

    // get view id for logs
    const IndexT viewId = mp.getViewId(camId);

    // find out with the LRU (Least Recently Used) strategy if the mipmap image is already in the cache
    // note: if new insertion in the cache, we need to replace a cached object with the new one
    int deviceMipmapId;
    const bool newInsertion = currentDeviceCache.mipmapCache.insert(camId, &deviceMipmapId);

    // check if the camera is already in cache
    if(!newInsertion)
    {
        LOG_X("Add mipmap image on device cache: already on cache (id: " << camId << ", view id: " << viewId << ").");
        return; // nothing to do
    }

    LOG_X("Add mipmap image on device cache (id: " << camId << ", view id: " << viewId << ").");

    // get image buffer
//    mvsUtils::ImagesCache<image::Image<image::RGBAfColor>>::ImgSharedPtr img = imageCache.getImg_sync(camId);
    // allocate the full size host-sided image buffer
    MTLSize imgSize = MTLSizeMake(mp.getOriginalWidth(camId), mp.getOriginalHeight(camId), 1);
    
//    DeviceTexture* imgTexture = [DeviceTexture new];
//    [_vCamTexturesCache insertObject:[imgTexture initWithSize:imgSize] atIndex:camId];
    
    DeviceBuffer* img_hmh = [DeviceBuffer new];
    
    assert(mp.imageBuffersCache[camId].size() == imgSize.width * imgSize.height * 4);
    
    auto* pDataFlipY = new uint8_t[imgSize.width * imgSize.height * 4];
    for(int i=0;i<imgSize.height; ++i) {
        int stride = imgSize.width*4;
        memcpy(pDataFlipY+stride*i, mp.imageBuffersCache[camId].data()+stride*(imgSize.height-i-1), stride);
    }
    [img_hmh initWithBytes:pDataFlipY size:imgSize elemSizeInBytes:sizeof(simd_uchar4)  elemType:@"uchar4"];
    delete [] pDataFlipY;
    
    
    // copy image from imageCache to CUDA host-side image buffer
    //TODO: cpu => gpu
//#pragma omp parallel for
//    for(int y = 0; y < imgSize.y(); ++y)
//    {
//        for(int x = 0; x < imgSize.x(); ++x)
//        {
//            const image::RGBAfColor& floatRGBA = (*img)(y, x);
//            CudaRGBA& cudaRGBA = img_hmh(x, y);
//
//#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
//            // explicit float to half conversion
//            cudaRGBA.x = half(floatRGBA.r() * 255.0f);
//            cudaRGBA.y = half(floatRGBA.g() * 255.0f);
//            cudaRGBA.z = half(floatRGBA.b() * 255.0f);
//            cudaRGBA.w = half(floatRGBA.a() * 255.0f);
//#else
//            cudaRGBA.x = floatRGBA.r() * 255.0f;
//            cudaRGBA.y = floatRGBA.g() * 255.0f;
//            cudaRGBA.z = floatRGBA.b() * 255.0f;
//            cudaRGBA.w = floatRGBA.a() * 255.0f;
//#endif
//        }
//    }

    DeviceMipmapImage& deviceMipmapImage = *(currentDeviceCache.mipmaps.at(deviceMipmapId));
    deviceMipmapImage.fill(img_hmh, minDownscale, maxDownscale); //TODO: USE MTLMipmap ?
}

void DeviceCache::addCameraParams(int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    // get current device cache
    SingleDeviceCache& currentDeviceCache = getCurrentDeviceCache();

    // get view id for logs
    const IndexT viewId = mp.getViewId(camId);

    // find out with the LRU (Least Recently Used) strategy if the camera parameters is already in the cache
    // note: if new insertion in the cache, we need to replace a cached object with the new one
    int deviceCameraParamsId;
    const bool newInsertion = currentDeviceCache.cameraParamCache.insert(CameraPair(camId, downscale), &deviceCameraParamsId);

    // check if the camera is already in cache
    if(!newInsertion)
    {
        LOG_X("Add camera parameters on device cache: already on cache (id: " << camId << ", view id: " << viewId << ", downscale: " << downscale << ").");
        return; // nothing to do
    }

    LOG_X("Add camera parameters on device cache (id: " << camId << ", view id: " << viewId << ", downscale: " << downscale << ").");

    {
        DeviceCameraParams params;
        fillCameraParameters(params,camId, downscale, mp);
        _vCamParams.emplace_back(params);
    }
}

const DeviceMipmapImage& DeviceCache::requestMipmapImage(int camId, const mvsUtils::MultiViewParams& mp)
{
    // get current device cache
    SingleDeviceCache& currentDeviceCache = getCurrentDeviceCache();

    // get view id for logs
    const IndexT viewId = mp.getViewId(camId);

    LOG_X("Request mipmap image on device cache (id: " << camId << ", view id: " << viewId << ").");

    // find out with the LRU (Least Recently Used) strategy if the mipmap image is already in the cache
    // note: if not found in cache we need to throw an error: in that case we insert an orphan id in the cache
    int deviceMipmapId;
    const bool notFound = currentDeviceCache.mipmapCache.insert(camId, &deviceMipmapId);

    // check if the mipmap image is in the cache
    if(notFound)
        ALICEVISION_THROW_ERROR("Request mipmap image on device cache: Not found (id: " << camId << ", view id: " << viewId << ").")

    // return the cached device mipmap image
    return *(currentDeviceCache.mipmaps.at(deviceMipmapId));
}

const int DeviceCache::requestCameraParamsId(int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    // get current device cache
    SingleDeviceCache& currentDeviceCache = getCurrentDeviceCache();

    // get view id for logs
    const IndexT viewId = mp.getViewId(camId);

    LOG_X("Request camera parameters on device cache (id: " << camId << ", view id: " << viewId << ", downscale: " << downscale << ").");

    // find out with the LRU (Least Recently Used) strategy if the camera parameters object is already in the cache
    // note: if not found in cache we need to throw an error: in that case we insert an orphan id in the cache
    int deviceCameraParamsId;
    const bool notFound = currentDeviceCache.cameraParamCache.insert(CameraPair(camId, downscale), &deviceCameraParamsId);

    // check if the camera is in the cache
    if(notFound)
        ALICEVISION_THROW_ERROR("Request camera parameters on device cache: Not found (id: " << camId << ", view id: " << viewId << ", downscale: " << downscale << ").")

    // return the cached camera parameters id
    return deviceCameraParamsId;
}

DeviceCameraParams const& DeviceCache::requestCameraParamsBuffer(int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    return _vCamParams[requestCameraParamsId(camId, downscale, mp)];
}

DeviceCameraParams const& DeviceCache::requestCameraParamsBuffer(int index)
{
    return _vCamParams[index];
}
} // namespace depthMap

