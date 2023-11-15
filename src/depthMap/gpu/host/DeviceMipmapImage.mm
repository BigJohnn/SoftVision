// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "DeviceMipmapImage.hpp"

#include <SoftVisionLog.h>
#include <numeric/numeric.hpp>


namespace depthMap {

DeviceMipmapImage::~DeviceMipmapImage()
{
    // destroy mipmapped array texture object
//    if(_textureObject != 0)
//      CHECK_CUDA_RETURN_ERROR_NOEXCEPT(cudaDestroyTextureObject(_textureObject));
//
//    // free mipmapped array
//    if(_mipmappedArray != nullptr)
//      CHECK_CUDA_RETURN_ERROR_NOEXCEPT(cudaFreeMipmappedArray(_mipmappedArray));
}

void DeviceMipmapImage::fill(DeviceBuffer* in_img_hmh, int minDownscale, int maxDownscale)
{
    // update private members
    _minDownscale = minDownscale;
    _maxDownscale = maxDownscale;
    _width  = [in_img_hmh getSize].width;
    _height  = [in_img_hmh getSize].height;
    _levels = log2(maxDownscale / minDownscale) + 1;

    DeviceTexture* texture = [DeviceTexture new];
    _texture = [texture initWithBuffer:in_img_hmh];
    
    // destroy previous texture object
//    if(_textureObject != 0)
//      CHECK_CUDA_RETURN_ERROR(cudaDestroyTextureObject(_textureObject));

    // destroy previous mipmapped array
//    if(_mipmappedArray != nullptr)
//      CHECK_CUDA_RETURN_ERROR(cudaFreeMipmappedArray(_mipmappedArray));

    // allocate the device-sided full-size input image buffer
//    auto img_dmpPtr = std::make_shared<CudaDeviceMemoryPitched<CudaRGBA, 2>>(in_img_hmh.getSize());

    // copy the host-sided full-size input image buffer onto the device-sided image buffer
//    img_dmpPtr->copyFrom(in_img_hmh);

    // downscale device-sided full-size input image buffer to min downscale
//    if(minDownscale > 1)
//    {
//        // create full-size input image buffer texture
////        CudaRGBATexture fullSizeImg(*img_dmpPtr);
//        id<MTLTexture> inputTexture;
//        {
//            MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
//
//            descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
//            descriptor.textureType      = MTLTextureType2D;
//
//            auto&& sz = img_dmpPtr->getSize();
//            descriptor.width            = sz.x();
//            descriptor.height           = sz.y();
//            descriptor.storageMode      = MTLStorageModePrivate;
//            descriptor.mipmapLevelCount = _levels;
//            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//            inputTexture = [device newTextureWithDescriptor:descriptor];
//
//            MTLRegion region = {
//                        { 0, 0, 0 },                   // MTLOrigin
//                        {descriptor.width, descriptor.height, 1} // MTLSize
//                    };
//
//            [inputTexture replaceRegion:region
//                       mipmapLevel:0
//                         withBytes:img_dmpPtr->getBuffer()
//                       bytesPerRow:img_dmpPtr->getUnpaddedBytesInRow()];
//        }
//
//        id<MTLTexture> downscaledTexture;
//        {
//            MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
//
//            descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
//            descriptor.textureType      = MTLTextureType2D;
//
//            auto&& sz = img_dmpPtr->getSize();
//            descriptor.width            = size_t(divideRoundUp(int(_width),  int(minDownscale)));
//            descriptor.height           = size_t(divideRoundUp(int(_height), int(minDownscale)));
//            descriptor.storageMode      = MTLStorageModePrivate;
//            descriptor.mipmapLevelCount = _levels;
//
//            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//            downscaledTexture = [device newTextureWithDescriptor:descriptor];
//        }
//
//
//        id <MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
//        [encoder generateMipmapsForTexture: inputTexture];
//
//        if (@available(iOS 13.0, *)) {
//            [encoder copyFromTexture: inputTexture
//                         sourceSlice: 0
//                         sourceLevel: log2(minDownscale)
//                           toTexture: downscaledTexture
//                    destinationSlice: 0
//                    destinationLevel: 0
//                          sliceCount: 1
//                          levelCount: 1];
//            [encoder generateMipmapsForTexture: downscaledTexture];//TODO: check
//        } else {
//            // Fallback on earlier versions
//        }
//
//        [encoder endEncoding];
//
//    }
}


float DeviceMipmapImage::getLevel(unsigned int downscale) const
{
  // check given downscale
  if(downscale < _minDownscale || downscale > _maxDownscale)
    ALICEVISION_THROW_ERROR("Cannot get device mipmap image level (downscale: " << downscale << ")");

  return log2(float(downscale) / float(_minDownscale));
}


MTLSize DeviceMipmapImage::getDimensions(unsigned int downscale) const
{
  // check given downscale
  if(downscale < _minDownscale || downscale > _maxDownscale)
    ALICEVISION_THROW_ERROR("Cannot get device mipmap image level dimensions (downscale: " << downscale << ")");

  return MTLSizeMake(divideRoundUp(int(_width), int(downscale)), divideRoundUp(int(_height), int(downscale)), 1);
}

} // namespace depthMap

