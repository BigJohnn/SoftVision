
#import <MetalKit/MetalKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface DeviceBuffer : NSObject

+(DeviceBuffer*) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type;
+(DeviceBuffer*) initWithBytes:(nonnull const void*)bytes size:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type;

-(void) copyFrom:(DeviceBuffer*)src;
-(id<MTLBuffer>) getBuffer;
-(id<MTLBuffer>) allocate:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type;
-(id<MTLBuffer>) initWithBytes:(nonnull const void*)bytes size:(MTLSize)size elemSizeInBytes:(int)nBytes elemType:(NSString*)type;
-(void*) getBufferPtr;
-(MTLSize) getSize;
-(int) getBufferLength;
-(int) getElemSize;

-(simd_float2) getVec2f:(int)x y:(int)y;
-(void) setVec2f:(simd_float2)val x:(int)x y:(int)y;

-(void) setVec1f:(float)val x:(int)x y:(int)y;

-(int) getBytesUpToDim:(int)dim;
-(NSString*) getElemType;
-(id<MTLTexture>) getDebugTexture:(int)sliceAlongZ;
-(id<MTLTexture>) getDebugTexture;

@end

NS_ASSUME_NONNULL_END

