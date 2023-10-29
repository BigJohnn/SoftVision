#import <Metal/Metal.h>

#include <depthMap/gpu/imageProcessing/deviceGaussianFilter.hpp>
#include <cstdlib>
#include <string>
namespace depthMap {

#define MAX_CONSTANT_GAUSS_SCALES   10
#define MAX_CONSTANT_GAUSS_MEM_SIZE 128

void createConstantGaussianArray(int cudaDeviceId, int scales)
{
    if(scales >= MAX_CONSTANT_GAUSS_SCALES)
    {
        throw std::runtime_error( "Programming error: too few scales pre-computed for Gaussian kernels. Enlarge and recompile." );
    }

    if(d_gaussianArrayInitialized.find(cudaDeviceId) != d_gaussianArrayInitialized.end())
        return;

    d_gaussianArrayInitialized.insert(cudaDeviceId);

    int*   h_gaussianArrayOffset;
    float* h_gaussianArray;
    
    //TODO： use malloc?
    h_gaussianArrayOffset = (int*)malloc(MAX_CONSTANT_GAUSS_SCALES * sizeof(int));
    h_gaussianArray = (float*)malloc(MAX_CONSTANT_GAUSS_MEM_SIZE * sizeof(float));
//    err = cudaMallocHost(&h_gaussianArrayOffset, MAX_CONSTANT_GAUSS_SCALES * sizeof(int));
//    THROW_ON_CUDA_ERROR(err, "Failed to allocate " << MAX_CONSTANT_GAUSS_SCALES * sizeof(int) << " of CUDA host memory.");
//
//    err = cudaMallocHost(&h_gaussianArray, MAX_CONSTANT_GAUSS_MEM_SIZE * sizeof(float));
//    THROW_ON_CUDA_ERROR(err, "Failed to allocate " << MAX_CONSTANT_GAUSS_MEM_SIZE * sizeof(float) << " of CUDA host memory.");
//
    int sumSizes = 0;

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        h_gaussianArrayOffset[scale] = sumSizes;
        const int radius = scale + 1;
        const int size = 2 * radius + 1;
        sumSizes += size;
    }

    if(sumSizes >= MAX_CONSTANT_GAUSS_MEM_SIZE)
    {
        throw std::runtime_error( "Programming error: too little memory allocated for "
            + std::to_string(MAX_CONSTANT_GAUSS_SCALES) + " Gaussian kernels. Enlarge and recompile." );
    }

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        const int radius = scale + 1;
        const float delta  = 1.0f;
        const int size   = 2 * radius + 1;

        for(int idx = 0; idx < size; idx++)
        {
            int x = idx - radius;
            h_gaussianArray[h_gaussianArrayOffset[scale]+idx] = expf(-(x * x) / (2 * delta * delta));
        }
    }
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> gaussianArrayOffsetVBO = [device newBufferWithBytes:h_gaussianArrayOffset length:MAX_CONSTANT_GAUSS_SCALES * sizeof(int) options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> gaussianArrayVBO = [device newBufferWithBytes:h_gaussianArray length:sumSizes * sizeof(float) options:MTLResourceStorageModeShared];
    
    free(h_gaussianArrayOffset);
    free(h_gaussianArray);
    
    NSError* error = nil;

    // Load the shader files with a .metal file extension in the project

    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    if (defaultLibrary == nil)
    {
        NSLog(@"Failed to find the default library.");
        return;
    }

    id<MTLFunction> getGaussfunc = [defaultLibrary newFunctionWithName:@"getGauss"]; // todo
    if (getGaussfunc == nil)
    {
        NSLog(@"Failed to find the getGauss function.");
        return ;
    }

    // Create a compute pipeline state object.
    id<MTLComputePipelineState> getGaussFuncPSO = [device newComputePipelineStateWithFunction: getGaussfunc error:&error];
    if (getGaussFuncPSO == nil)
    {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        NSLog(@"Failed to created pipeline state object, error %@.", error);
        return;
    }
    
    // create cuda array
//    _vertexBuffer = [_device newBufferWithBytes:vertexData
//                                         length:sizeof(vertexData)
//                                        options:MTLResourceStorageModeShared];
//
//    err = cudaMemcpyToSymbol( d_gaussianArrayOffset,
//                              h_gaussianArrayOffset,
//                              MAX_CONSTANT_GAUSS_SCALES * sizeof(int), 0, cudaMemcpyHostToDevice);
//
//    THROW_ON_CUDA_ERROR(err, "Failed to move Gaussian filter to symbol.");
//
//    err = cudaMemcpyToSymbol(d_gaussianArray,
//                             h_gaussianArray,
//                             sumSizes * sizeof(float), 0, cudaMemcpyHostToDevice);
//
//    THROW_ON_CUDA_ERROR(err, "Failed to move Gaussian filter to symbol." );
//
//    cudaFreeHost(h_gaussianArrayOffset);
//    cudaFreeHost(h_gaussianArray);

}


}
