#pragma once

#include <simd/simd.h>
namespace depthMap {

/**
 * @struct DeviceCameraParams
 * @brief Support class to maintain useful camera parameters in gpu memory.
 */
typedef struct DeviceCameraParams
{
    vector_float P[12];
    vector_float iP[9];
    vector_float R[9];
    vector_float iR[9];
    vector_float K[9];
    vector_float iK[9];
    vector_float3 C;
    vector_float3 XVect;
    vector_float3 YVect;
    vector_float3 ZVect;
} DeviceCameraParams;

// global / constant data structures

#define ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS 50 // CUDA constant memory is limited to 65K(100) TODO: check this, we use metal 32k(50)

extern __constant__ DeviceCameraParams constantCameraParametersArray_d[ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS];

} // namespace depthMap
