#pragma once

namespace depthMap {

/**
 * @struct DeviceCameraParams
 * @brief Support class to maintain useful camera parameters in gpu memory.
 */
struct DeviceCameraParams
{
    float P[12];
    float iP[9];
    float R[9];
    float iR[9];
    float K[9];
    float iK[9];
    float3 C;
    float3 XVect;
    float3 YVect;
    float3 ZVect;
};

// global / constant data structures

#define ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS 100 // CUDA constant memory is limited to 65K TODO: check this, we use metal

extern __constant__ DeviceCameraParams constantCameraParametersArray_d[ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS];

} // namespace depthMap
