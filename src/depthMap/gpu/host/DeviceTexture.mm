//
//  RGBATexture.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/8.
//

#import <depthMap/gpu/host/DeviceTexture.hpp>

@interface DeviceTexture()
@end

@implementation DeviceTexture

-(id<MTLTexture>) initWithSize:(MTLSize)size
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.textureType      = MTLTextureType2D;
    
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModePrivate;
    
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return [device newTextureWithDescriptor:descriptor];
}

-(id<MTLTexture>) initWithBuffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
    
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MTLRegion region = {
            { 0, 0, 0 },                   // MTLOrigin
            {descriptor.width, descriptor.height, 1} // MTLSize
        };
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:[buffer getBuffer].contents
               bytesPerRow:[buffer getBytesUpToDim:0]];
    return texture;
}

+(id<MTLTexture>) initWithFloatBuffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
    descriptor.pixelFormat = MTLPixelFormatR16Float;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
    
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MTLRegion region = {
            { 0, 0, 0 },                   // MTLOrigin
            {descriptor.width, descriptor.height, 1} // MTLSize
        };
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:[buffer getBuffer].contents
               bytesPerRow:[buffer getBytesUpToDim:0]];
    return texture;
}
@end

