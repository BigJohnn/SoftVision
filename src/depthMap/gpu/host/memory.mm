//
//  memory.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/10.
//

#import <depthMap/gpu/host/memory.hpp>

@implementation DeviceBuffer
{
    id<MTLBuffer> buffer;
    MTLSize sz;
    int elemSizeInBytes;
    int nBytesPerRow;
    int bufferLengthInBytes;
}

-(void) copyFrom:(DeviceBuffer*)src
{
//    id<MTLBlitCommandEncoder> encoder;
//    [encoder copyFromBuffer:[src getBuffer] sourceOffset:0 toBuffer:buffer destinationOffset:0 size:[src getBufferLength]];
    
    buffer = [src getBuffer]; // shallow copy
    sz = [src getSize];
    elemSizeInBytes = [src getElemSize];
    nBytesPerRow = [src getBytesUpToDim:0];
    bufferLengthInBytes = [src getBufferLength];
}

+(DeviceBuffer*) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes
{
    DeviceBuffer* buf = [DeviceBuffer new];
//    buf->buffer
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
           
    buf->nBytesPerRow = static_cast<int>(size.width) * nBytes;
    
    buf->bufferLengthInBytes = buf->nBytesPerRow * size.height * size.depth;
    
    buf->buffer = [device newBufferWithLength:nBytes
                                         options:MTLResourceStorageModeShared];
    
    buf->sz = size;
    
    buf->elemSizeInBytes = nBytes;
    return buf;
}

-(id<MTLBuffer>) getBuffer
{
    return buffer;
}

-(id<MTLBuffer>) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    nBytesPerRow = static_cast<int>(size.width) * nBytes;

    bufferLengthInBytes = nBytesPerRow * size.height * size.depth;

    buffer = [device newBufferWithLength:nBytesPerRow
                                         options:MTLResourceStorageModeShared];

    sz = size;

    elemSizeInBytes = nBytes;
    return buffer;
}

-(id<MTLBuffer>) initWithBytes:(nonnull const void*)bytes size:(MTLSize)size elemSizeInBytes:(int)nBytes
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    nBytesPerRow = static_cast<int>(size.width) * nBytes;

    bufferLengthInBytes = nBytesPerRow * size.height * size.depth;
    
    buffer = [device newBufferWithBytes:bytes length:bufferLengthInBytes options:MTLResourceStorageModeShared];
    
    sz = size;

    elemSizeInBytes = nBytes;
    
    return buffer;
}

-(simd_float2) getVec2f:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(simd_float2));
    return *((simd_float2*)buffer.contents + y * nBytesPerRow + x * sizeof(simd_float2));
}

-(void) setVec2f:(simd_float2)val x:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(simd_float2));
    *((simd_float2*)buffer.contents + y * nBytesPerRow + x * sizeof(simd_float2)) = val;
}

-(void) setVec1f:(float)val x:(int)x y:(int)y
{
    assert(elemSizeInBytes == sizeof(float));
    *((float*)buffer.contents + y * nBytesPerRow + x * sizeof(float)) = val;
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

@end
