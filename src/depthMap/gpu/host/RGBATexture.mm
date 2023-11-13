//
//  RGBATexture.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/8.
//

#import <Metal/Metal.h>

#include <depthMap/gpu/host/memory.hpp>

namespace depthMap{

CudaRGBATexture::CudaRGBATexture(CudaDeviceMemoryPitched<CudaRGBA, 2>& buffer_dmp)
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.textureType      = MTLTextureType2D;
    
    auto&& sz = buffer_dmp.getSize();
    descriptor.width            = sz.x();
    descriptor.height           = sz.y();
    descriptor.storageMode      = MTLStorageModePrivate;
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    
//    [buffer_dmp->getBuffer() mak]
    textureObj = (__bridge void*)[device newTextureWithDescriptor:descriptor];
}

}
