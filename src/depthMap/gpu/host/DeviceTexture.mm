//
//  RGBATexture.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/8.
//

#import <depthMap/gpu/host/DeviceTexture.hpp>
#import <depthMap/gpu/host/ComputePipeline.hpp>

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
    
//    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
    descriptor.mipmapLevelCount = 8;
    
    
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
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
    [encoder generateMipmapsForTexture: texture];
    [encoder endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    
    return texture;
}

+(id<MTLTexture>) initWithBuffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
//    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
    descriptor.mipmapLevelCount = 8;
    
    
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
    
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
    [encoder generateMipmapsForTexture: texture];
    [encoder endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
    
    return texture;
}

+(id<MTLTexture>) initWithFloatBuffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
    descriptor.pixelFormat = MTLPixelFormatR32Float;
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

+(id<MTLTexture>) initWithUint8Buffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
//    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.pixelFormat = MTLPixelFormatR8Unorm_sRGB;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
//    descriptor.mipmapLevelCount = 8;
    
    
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
    
//    ComputePipeline* pipeline = [ComputePipeline createPipeline];
//    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
//    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
//    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
//    [encoder generateMipmapsForTexture: texture];
//    [encoder endEncoding];
//    [cmdbuf commit];
//    [cmdbuf waitUntilCompleted];
    
    return texture;
}

+(id<MTLTexture>) initWithUint32Buffer:(DeviceBuffer*)buffer
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
//    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.pixelFormat = MTLPixelFormatR32Uint;
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
//    descriptor.mipmapLevelCount = 8;
    
    
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
    
//    ComputePipeline* pipeline = [ComputePipeline createPipeline];
//    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
//    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
//    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
//    [encoder generateMipmapsForTexture: texture];
//    [encoder endEncoding];
//    [cmdbuf commit];
//    [cmdbuf waitUntilCompleted];
    
    return texture;
}

+(id<MTLTexture>) initWithBuffer:(DeviceBuffer*)buffer pixelFormat:(NSString*)format
{
    MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];
    
//    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    if([format isEqualToString:@"TSim"]) {
        descriptor.pixelFormat = MTLPixelFormatR8Unorm;
    }
    else if([format isEqualToString:@"TSimAcc"]) {
        descriptor.pixelFormat = MTLPixelFormatR32Uint;
    }
    else if([format isEqualToString:@"float"] || [format isEqualToString:@"TSimRefine"]) {
        descriptor.pixelFormat = MTLPixelFormatR32Float;
    }
    else if([format isEqualToString:@"float2"]) {
        descriptor.pixelFormat = MTLPixelFormatRG32Float;
    }
    else if([format isEqualToString:@"float3"]) {
        descriptor.pixelFormat = MTLPixelFormatRGBA32Float;
    }
    
    descriptor.textureType      = MTLTextureType2D;
    
    MTLSize size = [buffer getSize];
    descriptor.width            = size.width;
    descriptor.height           = size.height;
    descriptor.storageMode      = MTLStorageModeShared;
//    descriptor.mipmapLevelCount = 8;
    
    
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
    
//    ComputePipeline* pipeline = [ComputePipeline createPipeline];
//    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
//    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
//    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
//    [encoder generateMipmapsForTexture: texture];
//    [encoder endEncoding];
//    [cmdbuf commit];
//    [cmdbuf waitUntilCompleted];
    
    return texture;
}


@end

