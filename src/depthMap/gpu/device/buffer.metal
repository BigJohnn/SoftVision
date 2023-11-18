//#pragma once

#include <depthMap/gpu/device/DeviceCameraParams.hpp>

//#include <depthMap/gpu/device/BufPtr.metal>
//#include <depthMap/BufPtr.hpp>

#include <metal_stdlib>

using namespace metal;

//#include "depthMap/gpu/device/DeviceCameraParams.hpp"

namespace depthMap {



/**
* @brief
* @param[int] ptr
* @param[int] spitch raw length of a 2D array in bytes
* @param[int] pitch raw length of a line in bytes
* @param[int] x
* @param[int] y
* @return
*/
template <typename T>
inline thread T* get3DBufferAt(device T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((thread T*)(((device char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
inline const thread T* get3DBufferAt(const device T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((const thread T*)(((const device char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
inline thread T* get3DBufferAt(device T* ptr, size_t spitch, size_t pitch, const device int3& v)
{
    return get3DBufferAt(ptr, spitch, pitch, v.x, v.y, v.z);
}

template <typename T>
inline const thread T* get3DBufferAt(const device T* ptr, size_t spitch, size_t pitch, const device int3& v)
{
    return get3DBufferAt(ptr, spitch, pitch, v.x, v.y, v.z);
}

inline float multi_fminf(float a, float b, float c)
{
  return min(min(a, b), c);
}

inline float multi_fminf(float a, float b, float c, float d)
{
  return min(min(min(a, b), c), d);
}


//#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
//
//inline float4 tex2D_float4(cudaTextureObject_t rc_tex, float x, float y)
//{
//#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_INTERPOLATION
//    // cudaReadNormalizedFloat
//    float4 a = tex2D<float4>(rc_tex, x, y);
//    return make_float4(a.x * 255.0f, a.y * 255.0f, a.z * 255.0f, a.w * 255.0f);
//#else
//    // cudaReadElementType
//    uchar4 a = tex2D<uchar4>(rc_tex, x, y);
//    return make_float4(a.x, a.y, a.z, a.w);
//#endif
//}
//
//#else
//
//inline float4 tex2D_float4(cudaTextureObject_t rc_tex, float x, float y)
//{
//    return tex2D<float4>(rc_tex, x, y);
//}
//
//#endif

} // namespace depthMap

