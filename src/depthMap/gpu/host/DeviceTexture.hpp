
#import <MetalKit/MetalKit.h>
#import <depthMap/gpu/host/memory.hpp>

NS_ASSUME_NONNULL_BEGIN

@interface DeviceTexture : NSObject

-(id<MTLTexture>) initWithSize:(MTLSize)size;
-(id<MTLTexture>) initWithBuffer:(DeviceBuffer*)buffer;
+(id<MTLTexture>) initWithFloatBuffer:(DeviceBuffer*)buffer;

@end

NS_ASSUME_NONNULL_END

