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


inline float2 project3DPoint(const device float* M3x4, const device float3& V)
{
    // without optimization
    // const float3 p = M3x4mulV3(M3x4, V);
    // return float2(p.x / p.z, p.y / p.z);

    float3 p = M3x4mulV3(M3x4, V);
    const float pzInv =  divide(1.0f, p.z);
    return float2(p.x * pzInv, p.y * pzInv);
}

inline void M3x3mulM3x3(device float* O3x3, const device float* A3x3, const device float* B3x3)
{
    O3x3[0] = A3x3[0] * B3x3[0] + A3x3[3] * B3x3[1] + A3x3[6] * B3x3[2];
    O3x3[3] = A3x3[0] * B3x3[3] + A3x3[3] * B3x3[4] + A3x3[6] * B3x3[5];
    O3x3[6] = A3x3[0] * B3x3[6] + A3x3[3] * B3x3[7] + A3x3[6] * B3x3[8];

    O3x3[1] = A3x3[1] * B3x3[0] + A3x3[4] * B3x3[1] + A3x3[7] * B3x3[2];
    O3x3[4] = A3x3[1] * B3x3[3] + A3x3[4] * B3x3[4] + A3x3[7] * B3x3[5];
    O3x3[7] = A3x3[1] * B3x3[6] + A3x3[4] * B3x3[7] + A3x3[7] * B3x3[8];

    O3x3[2] = A3x3[2] * B3x3[0] + A3x3[5] * B3x3[1] + A3x3[8] * B3x3[2];
    O3x3[5] = A3x3[2] * B3x3[3] + A3x3[5] * B3x3[4] + A3x3[8] * B3x3[5];
    O3x3[8] = A3x3[2] * B3x3[6] + A3x3[5] * B3x3[7] + A3x3[8] * B3x3[8];
}

inline void M3x3minusM3x3(device float* O3x3, device float* A3x3, device float* B3x3)
{
    O3x3[0] = A3x3[0] - B3x3[0];
    O3x3[1] = A3x3[1] - B3x3[1];
    O3x3[2] = A3x3[2] - B3x3[2];
    O3x3[3] = A3x3[3] - B3x3[3];
    O3x3[4] = A3x3[4] - B3x3[4];
    O3x3[5] = A3x3[5] - B3x3[5];
    O3x3[6] = A3x3[6] - B3x3[6];
    O3x3[7] = A3x3[7] - B3x3[7];
    O3x3[8] = A3x3[8] - B3x3[8];
}

inline void M3x3transpose(device float* O3x3, const device float* A3x3)
{
    O3x3[0] = A3x3[0];
    O3x3[1] = A3x3[3];
    O3x3[2] = A3x3[6];
    O3x3[3] = A3x3[1];
    O3x3[4] = A3x3[4];
    O3x3[5] = A3x3[7];
    O3x3[6] = A3x3[2];
    O3x3[7] = A3x3[5];
    O3x3[8] = A3x3[8];
}

inline void outerMultiply(device float3x3& O3x3, const device float3& a, const device float3& b)
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

inline float3 linePlaneIntersect(const device float3& linePoint,
                                            const device float3& lineVect,
                                            const device float3& planePoint,
                                            const device float3& planeNormal)
{
    const float k = (dot(planePoint, planeNormal) - dot(planeNormal, linePoint)) / dot(planeNormal, lineVect);
    return linePoint + lineVect * k;
}

inline float3 closestPointOnPlaneToPoint(const device float3& point, const device float3& planePoint, const device float3& planeNormalNormalized)
{
    return point - planeNormalNormalized * dot(planeNormalNormalized, point - planePoint);
}

inline float3 closestPointToLine3D(const device float3& point, const device float3& linePoint, const device float3& lineVectNormalized)
{
    return linePoint + lineVectNormalized * dot(lineVectNormalized, point - linePoint);
}

inline float pointLineDistance3D(const device float3& point, const device float3& linePoint, const device float3& lineVectNormalized)
{
    return length(cross(lineVectNormalized, linePoint - point));
}

// v1,v2 dot not have to be normalized
inline float angleBetwV1andV2(const device float3& iV1, const device float3& iV2)
{
    float3 V1 = iV1;
    normalize(V1);

    float3 V2 = iV2;
    normalize(V2);

    return abs(acos(V1.x * V2.x + V1.y * V2.y + V1.z * V2.z) / (M_PI_F / 180.0f));
}

inline float angleBetwABandAC(const device float3& A, const device float3& B, const device float3& C)
{
    float3 V1 = B - A;
    float3 V2 = C - A;

    normalize(V1);
    normalize(V2);

    const float x = float(V1.x * V2.x + V1.y * V2.y + V1.z * V2.z);
    float a = acos(x);
    a = isinf(a) ? 0.0 : a;
    return float(fabs(a) / (M_PI_F / 180.0));
}

/**
 * @brief Calculate the line segment PaPb that is the shortest route between two lines p1-p2 and p3-p4.
 *        Calculate also the values of mua and mub where:
 *          -> pa = p1 + mua (p2 - p1)
 *          -> pb = p3 + mub (p4 - p3)
 *
 * @note This a simple conversion to MATLAB of the C code posted by Paul Bourke at:
 *       http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/
 *       The author of this all too imperfect translation is Cristian Dima (csd@cmu.edu).
 *
 * @see https://web.archive.org/web/20060422045048/http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/
 */
inline float3 lineLineIntersect(device float* k,
                                device float* l,
                                device float3* lli1,
                                device float3* lli2,
                                device const float3& p1,
                                device const float3& p2,
                                device const float3& p3,
                                device const float3& p4)
{
    float d1343, d4321, d1321, d4343, d2121, denom, numer, p13[3], p43[3], p21[3], pa[3], pb[3], muab[2];

    p13[0] = p1.x - p3.x;
    p13[1] = p1.y - p3.y;
    p13[2] = p1.z - p3.z;

    p43[0] = p4.x - p3.x;
    p43[1] = p4.y - p3.y;
    p43[2] = p4.z - p3.z;

    /*
    if ((abs(p43[0])  < eps) & ...
        (abs(p43[1])  < eps) & ...
        (abs(p43[2])  < eps))
      error('Could not compute LineLineIntersect!');
    end
    */

    p21[0] = p2.x - p1.x;
    p21[1] = p2.y - p1.y;
    p21[2] = p2.z - p1.z;

    /*
    if ((abs(p21[0])  < eps) & ...
        (abs(p21[1])  < eps) & ...
        (abs(p21[2])  < eps))
      error('Could not compute LineLineIntersect!');
    end
    */

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

    denom = d2121 * d4343 - d4321 * d4321;

    /*
    if (abs(denom) < eps)
      error('Could not compute LineLineIntersect!');
    end
     */

    numer = d1343 * d4321 - d1321 * d4343;

    muab[0] = numer / denom;
    muab[1] = (d1343 + d4321 * muab[0]) / d4343;

    pa[0] = p1.x + muab[0] * p21[0];
    pa[1] = p1.y + muab[0] * p21[1];
    pa[2] = p1.z + muab[0] * p21[2];

    pb[0] = p3.x + muab[1] * p43[0];
    pb[1] = p3.y + muab[1] * p43[1];
    pb[2] = p3.z + muab[1] * p43[2];

    float3 S;
    S.x = (pa[0] + pb[0]) / 2.0;
    S.y = (pa[1] + pb[1]) / 2.0;
    S.z = (pa[2] + pb[2]) / 2.0;

    *k = muab[0];
    *l = muab[1];

    lli1->x = pa[0];
    lli1->y = pa[1];
    lli1->z = pa[2];

    lli2->x = pb[0];
    lli2->y = pb[1];
    lli2->z = pb[2];

    return S;
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

