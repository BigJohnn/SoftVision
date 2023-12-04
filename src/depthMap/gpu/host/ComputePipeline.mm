//
//  ComputePipeline.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/3.
//

#import <depthMap/gpu/host/ComputePipeline.hpp>
#import <objc/runtime.h>
@interface ComputePipeline()
@end

@implementation ComputePipeline

+(void) Exec:(MTLSize)gridSize ThreadgroupSize:(MTLSize)threadgroupSize KernelFuncName:(NSString*)kernelFuncName Args:(NSArray*)args
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    NSString* libraryName = @"sgm";

    NSBundle *bundle = [NSBundle bundleForClass:self.classForCoder];
    NSURL *bundleURL = [[bundle resourceURL] URLByAppendingPathComponent:@"metalshaders.bundle"];
    NSBundle *resourceBundle = [NSBundle bundleWithURL:bundleURL];
    NSURL *libraryURL = [resourceBundle URLForResource:libraryName
                                                    withExtension:@"metallib"];

    NSError *libraryError = nil;

    id <MTLLibrary> defaultLibrary = [device newLibraryWithURL:libraryURL
                                                  error:&libraryError];
    
    if (defaultLibrary == nil)
    {
        NSLog(@"Failed to find the default library.");
    }

    id<MTLFunction> func = [defaultLibrary newFunctionWithName:kernelFuncName];
    if (func == nil)
    {
        NSLog(@"Failed to find the %@ function.", kernelFuncName);
    }

    // Create a compute pipeline state object.
    NSError *error = nil;
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
    
    int texid = 0;
    int bufferid = 0;
    for(int i=0;i<args.count; ++i) {
        id elem = args[i];
        
//        NSLog(@"====%d", i);
        if([elem conformsToProtocol:@protocol(MTLBuffer)])
        {
            [computeEncoder setBuffer:elem offset:0 atIndex:bufferid++];
        }
        else if([elem conformsToProtocol:@protocol(MTLTexture)])
        {
            [computeEncoder setTexture:elem atIndex:texid++];
        }
        else if([elem isKindOfClass:[NSNumber class]]) {
            if ( strcmp([elem objCType], @encode(float)) == 0 ) {
                auto k = [elem floatValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
            }
            else if ( strcmp([elem objCType], @encode(int)) == 0 ) {
                auto k = [elem intValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
            }
            else if ( strcmp([elem objCType], @encode(long long)) == 0 ) {
                auto k = [elem unsignedIntValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
            }
            else if ( strcmp([elem objCType], @encode(bool)) == 0 ) {
                auto k = [elem boolValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
            }
            else if ( strcmp([elem objCType], @encode(double)) == 0 ) {
                auto k = [elem doubleValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
            }
            else {//TODO: check: uchar*, default case?
                auto k = [elem unsignedCharValue];
                [computeEncoder setBytes:&k length:sizeof(k) atIndex:bufferid++];
//                NSLog(@"==TODO: impl: elem encode is %s==",[elem objCType]);
            }
        }
        else if([elem isKindOfClass:[NSData class]]) {
            
            [computeEncoder setBytes:&elem length:[elem length] atIndex:bufferid++];
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
