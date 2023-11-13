
#import <MetalKit/MetalKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface DeviceBuffer : NSObject

-(id<MTLBuffer>) getBuffer;
-(id<MTLBuffer>) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes;
-(void*) getBufferPtr;
-(MTLSize) getSize;
-(int) getBufferLength;

-(simd_float2) getVec2f:(int)x y:(int)y;
-(void) setVec2f:(simd_float2)val x:(int)x y:(int)y;

-(void) setVec1f:(float)val x:(int)x y:(int)y;

@end

NS_ASSUME_NONNULL_END

