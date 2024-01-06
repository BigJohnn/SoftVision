// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "DeviceMipmapImage.hpp"

#include <SoftVisionLog.h>
#include <numeric/numeric.hpp>


namespace depthMap {

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

