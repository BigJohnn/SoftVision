//
//  ComputePipeline.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/3.
//

#import <depthMap/gpu/host/ComputePipeline.hpp>

@interface ComputePipeline()
@end

@implementation ComputePipeline

+(void) Exec:(MTLSize)gridSize ThreadgroupSize:(MTLSize)threadgroupSize KernelFuncName:(NSString*)kernelFuncName Args:(NSArray*)args
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    if (defaultLibrary == nil)
    {
        NSLog(@"Failed to find the default library.");
    }

    id<MTLFunction> func = [defaultLibrary newFunctionWithName:kernelFuncName];
    if (func == nil)
    {
        NSLog(@"Failed to find the depthThicknessMapSmoothThickness_kernel function.");
    }

    // Create a compute pipeline state object.
    NSError *error = [NSError new];
    id<MTLComputePipelineState> funcPSO = [device newComputePipelineStateWithFunction: func error:&error];
    if (funcPSO == nil)
    {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        NSLog(@"Failed to created pipeline state object, error %@.", error);
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (commandQueue == nil)
    {
        NSLog(@"Failed to find the command queue.");
    }
    
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:funcPSO];
    
    for(int i=0;i<args.count; ++i) {
        id elem = args[i];
        
        if([elem isKindOfClass:[NSNumber class]] || [elem isKindOfClass:[NSData class]]) {
            [computeEncoder setBytes:&elem length:sizeof(elem) atIndex:i];
        }
        else if([elem isKindOfClass:[NSObject class]]) { // TODO: check 如何判断这个NSOBject是MTLBuffer 或者MTLTexture
            [computeEncoder setBuffer:elem offset:0 atIndex:i];
        }
    }
    
    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];
}

@end
