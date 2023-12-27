//
//  memory.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/10.
//

#import <depthMap/gpu/host/memory.hpp>
#import <depthMap/gpu/host/DeviceTexture.hpp>
#import <depthMap/gpu/host/ComputePipeline.hpp>
#include <SoftVisionLog.h>

@implementation DeviceBuffer
{
    id<MTLBuffer> buffer;
    MTLSize sz;
    int elemSizeInBytes;
    int nBytesPerRow;
    int bufferLengthInBytes;
    NSString* elemType;
}

-(void) copyFrom:(DeviceBuffer*)src
{
    ComputePipeline* pipeline = [ComputePipeline createPipeline];
    id<MTLCommandQueue> queue =  [pipeline getCommandQueue];
    id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
    id <MTLBlitCommandEncoder> encoder = [cmdbuf blitCommandEncoder];
    [encoder copyFromBuffer:[src getBuffer] sourceOffset:0 toBuffer:buffer destinationOffset:0 size:[src getBufferLength]];
    [encoder endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

//    buffer = [src getBuffer]; // shallow copy
    sz = [src getSize];
    elemSizeInBytes = [src getElemSize];
    nBytesPerRow = [src getBytesUpToDim:0];
//    bufferLengthInBytes = [src getBufferLength];
    elemType = src->elemType;
}

+(DeviceBuffer*) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type
{
    DeviceBuffer* buf = [DeviceBuffer new];
    //    buf->buffer
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    buf->nBytesPerRow = static_cast<int>(size.width) * nBytes;
    
    buf->bufferLengthInBytes = buf->nBytesPerRow * size.height * size.depth;
    
    buf->buffer = [device newBufferWithLength:buf->bufferLengthInBytes
                                      options:MTLResourceStorageModeShared];
    
    buf->sz = size;
    
    buf->elemSizeInBytes = nBytes;
    
    buf->elemType = type;
    
    return buf;
}

-(id<MTLBuffer>) getBuffer
{
    return buffer;
}

-(id<MTLBuffer>) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    nBytesPerRow = static_cast<int>(size.width) * nBytes;
    
    bufferLengthInBytes = nBytesPerRow * size.height * size.depth;
    
    buffer = [device newBufferWithLength:bufferLengthInBytes
                                 options:MTLResourceStorageModeShared];
    
    sz = size;
    
    elemSizeInBytes = nBytes;
    
    elemType = type;
    
    return buffer;
}

-(id<MTLBuffer>) initWithBytes:(nonnull const void*)bytes size:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    nBytesPerRow = static_cast<int>(size.width) * nBytes;
    
    bufferLengthInBytes = nBytesPerRow * size.height * size.depth;
    
    buffer = [device newBufferWithBytes:bytes length:bufferLengthInBytes options:MTLResourceStorageModeShared];
    
    sz = size;
    
    elemSizeInBytes = nBytes;
    
    elemType = type;
    
    return buffer;
}

-(NSString*) getElemType
{
    return elemType;
}

+(DeviceBuffer*) initWithBytes:(nonnull const void*)bytes size:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type
{
    DeviceBuffer* buf = [DeviceBuffer new];
    [buf initWithBytes:bytes size:size elemSizeInBytes:nBytes elemType:type];
    return buf;
}

-(simd_float2) getVec2f:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(simd_float2));
    return *((simd_float2*)((char*)buffer.contents + y * nBytesPerRow + x * sizeof(simd_float2)));
}

-(void) setVec2f:(simd_float2)val x:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(simd_float2));
    *((simd_float2*)((char*)buffer.contents + y * nBytesPerRow + x * sizeof(simd_float2))) = val;
}

-(void) setVec1f:(float)val x:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(float));
    *((float*)((char*)buffer.contents + y * nBytesPerRow + x * sizeof(float))) = val;
}

-(void*) getBufferPtr
{
    return buffer.contents;
}

-(int) getBufferLength
{
    return bufferLengthInBytes;
}

-(int) getElemSize
{
    return elemSizeInBytes;
}

-(int) getBytesUpToDim:(int)dim
{
    int prod = nBytesPerRow;
    if(dim > 0) {
        prod *= sz.height;
        if(dim > 1) {
            prod *= sz.depth;
        }
    }
    return prod;
}

-(MTLSize) getSize
{
    return sz;
}

-(id<MTLTexture>) getDebugTexture:(int)sliceAlongZ
{
    if(sliceAlongZ >= sz.depth) {
        LOG_ERROR("z index out of range!");
        return nil;
    }
    DeviceBuffer* buff = [DeviceBuffer initWithBytes:((uint8_t*)buffer.contents + sz.width * sz.height * sliceAlongZ * elemSizeInBytes) size:MTLSizeMake(sz.width, sz.height, 1) elemSizeInBytes:elemSizeInBytes elemType:elemType];
    return [DeviceTexture initWithBuffer:buff pixelFormat:elemType];
}

-(id<MTLTexture>) getDebugTexture
{
    return [self getDebugTexture:0];
}

@end
