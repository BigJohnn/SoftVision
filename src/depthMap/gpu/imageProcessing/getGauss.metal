//
//  getGauss.metal
//  SoftVision
//
//  Created by HouPeihong on 2023/10/28.
//

#include <metal_stdlib>
using namespace metal;

kernel void getGauss(device const int* d_gaussianArrayOffset,
                       device const float* d_gaussianArray,
                     device float* o_gauss,
                       uint index [[thread_position_in_grid]],
                     constant int& scale [[buffer(0)]]) //TODO: check
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    o_gauss[index] =  d_gaussianArray[d_gaussianArrayOffset[scale] + index];
}
