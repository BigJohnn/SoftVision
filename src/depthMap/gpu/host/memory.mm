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
    int nBytes;
}

-(id<MTLBuffer>) getBuffer
{
    return buffer;
}

-(id<MTLBuffer>) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
           
    nBytesPerRow = static_cast<int>(size.width) * nBytes;
    
    nBytes = nBytesPerRow * size.height * size.depth;
    
    buffer = [device newBufferWithLength:nBytes
                                         options:MTLResourceStorageModeShared];
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
    return nBytes;
}


@end
