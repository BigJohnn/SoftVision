#pragma once
#include <metal_stdlib>
using namespace metal;

// mn MATRIX ADDRESSING: mxy = x*n+y (x-row,y-col), (m-number of rows, n-number of columns)


namespace depthMap {

//inline uchar4 float4_to_uchar4(const device float4& a)
//{
//    return uchar4((unsigned char)a.x, (unsigned char)a.y, (unsigned char)a.z, (unsigned char)a.w);
//}
//
//inline float4 uchar4_to_float4(const device uchar4& a)
//{
//    return float4((float)a.x, (float)a.y, (float)a.z, (float)a.w);
//}

//inline float dot(const device float3& a, const device float3& b)
//{
//    return a.x * b.x + a.y * b.y + a.z * b.z;
//}

//inline float dot(const float2& a, const float2& b)
//{
//    return a.x * b.x + a.y * b.y;
//}

//inline float size(const device float3& a)
//{
//    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
//}
//
//inline float size(const device float2& a)
//{
//    return sqrt(a.x * a.x + a.y * a.y);
//}
//
//inline float dist(const device float3& a, const device float3& b)
//{
//    return size(a - b);
//}
//
//inline float dist(const device float2& a, const device float2& b)
//{
//    return size(a - b);
//}

//inline float3 cross(const device float3& a, const device float3& b)
//{
//    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
//}

//inline float3 M3x3mulV3( const device float* M3x3, const device float3& V)
//{
//    return float3(M3x3[0] * V.x + M3x3[3] * V.y + M3x3[6] * V.z,
//                       M3x3[1] * V.x + M3x3[4] * V.y + M3x3[7] * V.z,
//                       M3x3[2] * V.x + M3x3[5] * V.y + M3x3[8] * V.z);
//}

//inline float3 M3x3mulV2( const device float* M3x3, const device float2& V)
//{
//    return float3(M3x3[0] * V.x + M3x3[3] * V.y + M3x3[6],
//                       M3x3[1] * V.x + M3x3[4] * V.y + M3x3[7],
//                       M3x3[2] * V.x + M3x3[5] * V.y + M3x3[8]);
//}

//inline float3 M3x4mulV3(const device float* M3x4, const device float3& V)
//{
//    return float3(M3x4[0] * V.x + M3x4[3] * V.y + M3x4[6] * V.z + M3x4[9],
//                       M3x4[1] * V.x + M3x4[4] * V.y + M3x4[7] * V.z + M3x4[10],
//                       M3x4[2] * V.x + M3x4[5] * V.y + M3x4[8] * V.z + M3x4[11]);
//}

//inline float2 V2M3x3mulV2(device float* M3x3, device float2& V)
//{
//    const float d = M3x3[2] * V.x + M3x3[5] * V.y + M3x3[8];
//    return float2((M3x3[0] * V.x + M3x3[3] * V.y + M3x3[6]) / d, (M3x3[1] * V.x + M3x3[4] * V.y + M3x3[7]) / d);
//}




//inline void M3x3mulM3x3(device float* O3x3, const device float* A3x3, const device float* B3x3)
//{
//    O3x3[0] = A3x3[0] * B3x3[0] + A3x3[3] * B3x3[1] + A3x3[6] * B3x3[2];
//    O3x3[3] = A3x3[0] * B3x3[3] + A3x3[3] * B3x3[4] + A3x3[6] * B3x3[5];
//    O3x3[6] = A3x3[0] * B3x3[6] + A3x3[3] * B3x3[7] + A3x3[6] * B3x3[8];
//
//    O3x3[1] = A3x3[1] * B3x3[0] + A3x3[4] * B3x3[1] + A3x3[7] * B3x3[2];
//    O3x3[4] = A3x3[1] * B3x3[3] + A3x3[4] * B3x3[4] + A3x3[7] * B3x3[5];
//    O3x3[7] = A3x3[1] * B3x3[6] + A3x3[4] * B3x3[7] + A3x3[7] * B3x3[8];
//
//    O3x3[2] = A3x3[2] * B3x3[0] + A3x3[5] * B3x3[1] + A3x3[8] * B3x3[2];
//    O3x3[5] = A3x3[2] * B3x3[3] + A3x3[5] * B3x3[4] + A3x3[8] * B3x3[5];
//    O3x3[8] = A3x3[2] * B3x3[6] + A3x3[5] * B3x3[7] + A3x3[8] * B3x3[8];
//}
//
//inline void M3x3minusM3x3(device float* O3x3, device float* A3x3, device float* B3x3)
//{
//    O3x3[0] = A3x3[0] - B3x3[0];
//    O3x3[1] = A3x3[1] - B3x3[1];
//    O3x3[2] = A3x3[2] - B3x3[2];
//    O3x3[3] = A3x3[3] - B3x3[3];
//    O3x3[4] = A3x3[4] - B3x3[4];
//    O3x3[5] = A3x3[5] - B3x3[5];
//    O3x3[6] = A3x3[6] - B3x3[6];
//    O3x3[7] = A3x3[7] - B3x3[7];
//    O3x3[8] = A3x3[8] - B3x3[8];
//}
//
//inline void M3x3transpose(device float* O3x3, const device float* A3x3)
//{
//    O3x3[0] = A3x3[0];
//    O3x3[1] = A3x3[3];
//    O3x3[2] = A3x3[6];
//    O3x3[3] = A3x3[1];
//    O3x3[4] = A3x3[4];
//    O3x3[5] = A3x3[7];
//    O3x3[6] = A3x3[2];
//    O3x3[7] = A3x3[5];
//    O3x3[8] = A3x3[8];
//}

inline void outerMultiply(thread float3x3& O3x3, const thread float3& a, const thread float3& b)
{
    O3x3[0] = a.x * b.x;
    O3x3[3] = a.x * b.y;
    O3x3[6] = a.x * b.z;
    O3x3[1] = a.y * b.x;
    O3x3[4] = a.y * b.y;
    O3x3[7] = a.y * b.z;
    O3x3[2] = a.z * b.x;
    O3x3[5] = a.z * b.y;
    O3x3[8] = a.z * b.z;
}

inline float3 linePlaneIntersect(device const float3& linePoint,
                                            const thread float3& lineVect,
                                            const thread float3& planePoint,
                                            const device float3& planeNormal)
{
    const float k = (dot(planePoint, planeNormal) - dot(planeNormal, linePoint)) / dot(planeNormal, lineVect);
    return linePoint + lineVect * k;
}

inline float3 closestPointOnPlaneToPoint(const thread float3& point, const thread float3& planePoint, const thread float3& planeNormalNormalized)
{
    return point - planeNormalNormalized * dot(planeNormalNormalized, point - planePoint);
}

inline float3 closestPointToLine3D(const thread float3& point, const thread float3& linePoint, const thread float3& lineVectNormalized)
{
    return linePoint + lineVectNormalized * dot(lineVectNormalized, point - linePoint);
}



// v1,v2 dot not have to be normalized
inline float angleBetwV1andV2(const device float3& iV1, const device float3& iV2)
{
    float3 V1 = iV1;
    V1 = normalize(V1);

    float3 V2 = iV2;
    V2 = normalize(V2);

    return abs(acos(V1.x * V2.x + V1.y * V2.y + V1.z * V2.z) / (M_PI_F / 180.0f));
}

inline float angleBetwABandAC(const device float3& A, const device float3& B, const device float3& C)
{
    float3 V1 = B - A;
    float3 V2 = C - A;

    V1 = normalize(V1);
    V2 = normalize(V2);

    const float x = float(V1.x * V2.x + V1.y * V2.y + V1.z * V2.z);
    float a = acos(x);
    a = isinf(a) ? 0.0 : a;
    return float(fabs(a) / (M_PI_F / 180.0));
}


/**
 * @brief Sigmoid function filtering
 * @note f(x) = min + (max-min) * \frac{1}{1 + e^{10 * (x - mid) / width}}
 * @see https://www.desmos.com/calculator/1qvampwbyx
 */
inline float sigmoid(float zeroVal, float endVal, float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal) * (1.0f / (1.0f + exp(10.0f * ((xval - sigMid) / sigwidth))));
}

/**
 * @brief Sigmoid function filtering
 * @note f(x) = min + (max-min) * \frac{1}{1 + e^{10 * (mid - x) / width}}
 */
inline float sigmoid2(float zeroVal, float endVal, float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal) * (1.0f / (1.0f + exp(10.0f * ((sigMid - xval) / sigwidth))));
}

} // namespace depthMap

