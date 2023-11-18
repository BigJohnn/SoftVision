//#include <depthMap/gpu/device/buffer.metal>
#include <depthMap/gpu/device/color.metal>
#include <depthMap/gpu/device/matrix.metal>
#include <depthMap/gpu/device/SimStat.metal>

#include <depthMap/gpu/device/DeviceCameraParams.hpp>
#include <depthMap/gpu/device/DevicePatchPattern.hpp>

#include <metal_stdlib>
using namespace metal;

#define INF_F -2.0f

namespace depthMap {

struct Patch
{
    float3 p; //< 3d point
    float3 n; //< normal
    float3 x; //< x axis
    float3 y; //< y axis
    float d;  //< pixel size
};

inline float2 project3DPoint(device const float4x3& M3x4, thread const float3& V)
{
    // without optimization
    // const float3 p = M3x4mulV3(M3x4, V);
    // return float2(p.x / p.z, p.y / p.z);

    float3 p = (M3x4 * float4(V, 1.0f)).xyz;
    const float pzInv =  divide(1.0f, p.z);
    return float2(p.x * pzInv, p.y * pzInv);
}
    
inline void rotPointAroundVect(thread float3& out, thread float3& X, thread float3& vect, int angle)
{
    float ux, uy, uz, vx, vy, vz, wx, wy, wz, sa, ca, x, y, z, u, v, w;

    float sizeX = sqrt(dot(X, X));
    x = X.x / sizeX;
    y = X.y / sizeX;
    z = X.z / sizeX;
    u = vect.x;
    v = vect.y;
    w = vect.z;

    /*Rotate the point (x,y,z) around the vector (u,v,w)*/
    ux = u * x;
    uy = u * y;
    uz = u * z;
    vx = v * x;
    vy = v * y;
    vz = v * z;
    wx = w * x;
    wy = w * y;
    wz = w * z;
    sa = sin((float)angle * (M_PI_F / 180.0f));
    ca = cos((float)angle * (M_PI_F / 180.0f));
    x = u * (ux + vy + wz) + (x * (v * v + w * w) - u * (vy + wz)) * ca + (-wy + vz) * sa;
    y = v * (ux + vy + wz) + (y * (u * u + w * w) - v * (ux + wz)) * ca + (wx - uz) * sa;
    z = w * (ux + vy + wz) + (z * (u * u + v * v) - w * (ux + vy)) * ca + (-vx + uy) * sa;

    u = sqrt(x * x + y * y + z * z);
    x /= u;
    y /= u;
    z /= u;

    out.x = x * sizeX;
    out.y = y * sizeX;
    out.z = z * sizeX;
}

inline void rotatePatch(thread Patch& ptch, int rx, int ry)
{
    float3 n, y, x;

    // rotate patch around x axis by angle rx
    rotPointAroundVect(n, ptch.n, ptch.x, rx);
    rotPointAroundVect(y, ptch.y, ptch.x, rx);
    ptch.n = n;
    ptch.y = y;

    // rotate new patch around y axis by angle ry
    rotPointAroundVect(n, ptch.n, ptch.y, ry);
    rotPointAroundVect(x, ptch.x, ptch.y, ry);
    ptch.n = n;
    ptch.x = x;
}

inline void movePatch(thread Patch& ptch, int pt)
{
    // float3 v = ptch.p-rC;
    // normalize(v);
    float3 v = ptch.n;

    float d = ptch.d * (float)pt;
    float3 p = ptch.p + v * d;
    ptch.p = p;
}

inline void computeRotCS(thread float3& xax, thread float3& yax, thread float3& n)
{
    xax.x = -n.y + n.z; // get any cross product
    xax.y = +n.x + n.z;
    xax.z = -n.x - n.y;
    if(fabs(xax.x) < 0.0000001f && fabs(xax.y) < 0.0000001f && fabs(xax.z) < 0.0000001f)
    {
        xax.x = -n.y - n.z; // get any cross product (complementar)
        xax.y = +n.x - n.z;
        xax.z = +n.x + n.y;
    };
    xax = normalize(xax);
    yax = cross(n, xax);
}

inline void computeRotCSEpip(thread Patch& ptch,
                             device const DeviceCameraParams& rcDeviceCamParams,
                             device const DeviceCameraParams& tcDeviceCamParams)
{
    // Vector from the reference camera to the 3d point
    float3 v1 = rcDeviceCamParams.C - ptch.p;
    // Vector from the target camera to the 3d point
    float3 v2 = tcDeviceCamParams.C - ptch.p;
    v1 = normalize(v1);
    v2 = normalize(v2);

    // y has to be ortogonal to the epipolar plane
    // n has to be on the epipolar plane
    // x has to be on the epipolar plane

    ptch.y = cross(v1, v2);
    ptch.y = normalize(ptch.y); // TODO: v1 & v2 are already normalized

    ptch.n = (v1 + v2) / 2.0f; // IMPORTANT !!!
    ptch.n = normalize(ptch.n); // TODO: v1 & v2 are already normalized
    // ptch.n = sg_s_r.ZVect; //IMPORTANT !!!

    ptch.x = cross(ptch.y, ptch.n);
    ptch.x = normalize(ptch.x);
}

inline float pointLineDistance3D(const thread float3& point, const device float3& linePoint, const thread float3& lineVectNormalized)
{
    return length(cross(lineVectNormalized, linePoint - point));
}
    
inline float computePixSize(const device DeviceCameraParams& deviceCamParams, thread const float3& p)
{
    const float2 rp = project3DPoint(deviceCamParams.P, p);
    const float2 rp1 = rp + float2(1.0f, 0.0f);
    
//    float3 refvect = M3x3mulV2(deviceCamParams.iP, rp1);
    float3 refvect = deviceCamParams.iP * float3(rp1, 1.0f);
    refvect = normalize(refvect);
    return pointLineDistance3D(p, deviceCamParams.C, refvect);
}

inline void getPixelFor3DPoint(thread float2& out, thread const DeviceCameraParams& deviceCamParams, thread const float3& X)
{
    const float3 p = (deviceCamParams.P * float4(X, 1.0f)).xyz;

    if(p.z <= 0.0f)
        out = float2(-1.0f, -1.0f);
    else
        out = float2(p.x / p.z, p.y / p.z);
}

inline float3 get3DPointForPixelAndFrontoParellePlaneRC(device const DeviceCameraParams& deviceCamParams, thread const float2& pix, float fpPlaneDepth)
{
    const float3 planep = deviceCamParams.C + deviceCamParams.ZVect * fpPlaneDepth;
    float3 v = deviceCamParams.iP * float3(pix, 1.0f);
    v = normalize(v);
    return linePlaneIntersect(deviceCamParams.C, v, planep, deviceCamParams.ZVect);
}

inline float3 get3DPointForPixelAndDepthFromRC(thread const DeviceCameraParams& deviceCamParams, thread const float2& pix, float depth)
{
    float3 rpv = deviceCamParams.iP * float3(pix, 1.0f);
    rpv = normalize(rpv);
    return deviceCamParams.C + rpv * depth;
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
inline float3 lineLineIntersect(thread float* k,
                                thread float* l,
                                thread float3* lli1,
                                thread float3* lli2,
                                thread const float3& p1,
                                thread const float3& p2,
                                thread const float3& p3,
                                thread const float3& p4)
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

inline float3 triangulateMatchRef(thread const DeviceCameraParams& rcDeviceCamParams,
                                  thread            const DeviceCameraParams& tcDeviceCamParams,
                                  thread            const float2& refpix,
                                  thread            const float2& tarpix)
{
    float3 refvect = rcDeviceCamParams.iP * float3(refpix, 1.0f);
    refvect = normalize(refvect);
    const float3 refpoint = refvect + rcDeviceCamParams.C;

    float3 tarvect = tcDeviceCamParams.iP * float3(tarpix, 1.0f);
    tarvect = normalize(tarvect);
    const float3 tarpoint = tarvect + tcDeviceCamParams.C;

    float k, l;
    float3 lli1, lli2;

    lineLineIntersect(&k, &l, &lli1, &lli2, rcDeviceCamParams.C, refpoint, tcDeviceCamParams.C, tarpoint);

    return rcDeviceCamParams.C + refvect * k;
}

/**
 * @brief Subpixel refine by Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation,
 *        and Occlusion Handling Qingxiong pami08.
 *
 * @note Quadratic polynomial interpolation is used to approximate the cost function between
 *       three discrete depth candidates: d, dA, and dB.
 *
 * @see https://pubmed.ncbi.nlm.nih.gov/19147877/
 *
 * @param[in] depths the 3 depths candidates (depth-1, depth, depth+1)
 * @param[in] sims the similarity of the 3 depths candidates
 *
 * @return refined depth value
 */
inline float refineDepthSubPixel(thread const float3& depths, thread const float3& sims)
{
    // TODO: get formula back from paper as it has been lost by encoding.
    // d is the discrete depth with the minimal cost, dA ? d A 1, and dB ? d B 1. The cost function is approximated as
    // f?x? ? ax2 B bx B c.

    float simM1 = sims.x;
    float sim = sims.y;
    float simP1 = sims.z;
    simM1 = (simM1 + 1.0f) / 2.0f;
    sim = (sim + 1.0f) / 2.0f;
    simP1 = (simP1 + 1.0f) / 2.0f;

    // sim is supposed to be the best one (so the smallest one)
    if((simM1 < sim) || (simP1 < sim))
        return depths.y; // return the input

    float dispStep = -((simP1 - simM1) / (2.0f * (simP1 + simM1 - 2.0f * sim)));

    float floatDepthM1 = depths.x;
    float floatDepthP1 = depths.z;

    //-1 : floatDepthM1
    // 0 : floatDepth
    //+1 : floatDepthP1
    // linear function fit
    // f(x)=a*x+b
    // floatDepthM1=-a+b
    // floatDepthP1= a+b
    // a = b - floatDepthM1
    // floatDepthP1=2*b-floatDepthM1
    float b = (floatDepthP1 + floatDepthM1) / 2.0f;
    float a = b - floatDepthM1;

    float interpDepth = a * dispStep + b;

    // Ensure that the interpolated value is isfinite  (i.e. neither infinite nor NaN)
    if(!isfinite(interpDepth) || interpDepth <= 0.0f)
        return depths.y; // return the input

    return interpDepth;
}

inline void computeRcTcMipmapLevels(thread float& out_rcMipmapLevel,
                                    thread            float& out_tcMipmapLevel,
                                    device            const float& mipmapLevel,
                                    device            const DeviceCameraParams& rcDeviceCamParams,
                                    device            const DeviceCameraParams& tcDeviceCamParams,
                                    thread            const float2& rp0,
                                    thread            const float2& tp0,
                                    thread            const float3& p0)
{
    // get p0 depth from the R camera
    const float rcDepth = length(rcDeviceCamParams.C - p0);

    // get p0 depth from the T camera
    const float tcDepth = length(tcDeviceCamParams.C - p0);

    // get R p0 corresponding pixel + 1x
    const float2 rp1 = rp0 + float2(1.f, 0.f);

    // get T p0 corresponding pixel + 1x
    const float2 tp1 = tp0 + float2(1.f, 0.f);

    // get rp1 3d point
    float3 rpv = (rcDeviceCamParams.iP * float3(rp1, 1.0f));
    rpv = normalize(rpv);
    const float3 prp1 = rcDeviceCamParams.C + rpv * rcDepth;

    // get tp1 3d point
    float3 tpv = (tcDeviceCamParams.iP * float3(tp1, 1.0f));
    tpv = normalize(tpv);
    const float3 ptp1 = tcDeviceCamParams.C + tpv * tcDepth;

    // compute 3d distance between p0 and rp1 3d point
    const float rcDist = distance(p0, prp1);

    // compute 3d distance between p0 and tp1 3d point
    const float tcDist = distance(p0, ptp1);

    // compute Rc/Tc distance factor
    const float distFactor = rcDist / tcDist;

    // set output R and T mipmap level
    if(distFactor < 1.f)
    {
        // T camera has a lower resolution (1 Rc pixSize < 1 Tc pixSize)
        out_tcMipmapLevel = mipmapLevel - log2(1.f / distFactor);

        if(out_tcMipmapLevel < 0.f)
        {
          out_rcMipmapLevel = mipmapLevel + abs(out_tcMipmapLevel);
          out_tcMipmapLevel = 0.f;
        }
    }
    else
    {
        // T camera has a higher resolution (1 Rc pixSize > 1 Tc pixSize)
        out_rcMipmapLevel = mipmapLevel;
        out_tcMipmapLevel = mipmapLevel + log2(distFactor);
    }
}

inline int angleBetwUnitV1andUnitV2(device float3& V1, device float3& V2)
{
    return (int)abs(acos(V1.x * V2.x + V1.y * V2.y + V1.z * V2.z) / (M_PI_F / 180.0f));
}

/*
inline float getRefCamPixSize(Patch &ptch)
{
        float2 rp = project3DPoint(sg_s_r.P,ptch.p);

        float minstep=10000000.0f;
        for (int i=0;i<4;i++) {
                float2 pix = rp;
                if (i==0) {pix.x += 1.0f;};
                if (i==1) {pix.x -= 1.0f;};
                if (i==2) {pix.y += 1.0f;};
                if (i==3) {pix.y -= 1.0f;};
                float3 vect = M3x3mulV2(sg_s_r.iP,pix);
                float3 lpi = linePlaneIntersect(sg_s_r.C, vect, ptch.p, ptch.n);
                float step = distance(lpi,ptch.p);
                minstep = fminf(minstep,step);
        };

        return minstep;
}

inline float getTarCamPixSize(Patch &ptch)
{
        float2 tp = project3DPoint(sg_s_t.P,ptch.p);

        float minstep=10000000.0f;
        for (int i=0;i<4;i++) {
                float2 pix = tp;
                if (i==0) {pix.x += 1.0f;};
                if (i==1) {pix.x -= 1.0f;};
                if (i==2) {pix.y += 1.0f;};
                if (i==3) {pix.y -= 1.0f;};
                float3 vect = M3x3mulV2(sg_s_t.iP,pix);
                float3 lpi = linePlaneIntersect(sg_s_t.C, vect, ptch.p, ptch.n);
                float step = distance(lpi,ptch.p);
                minstep = fminf(minstep,step);
        };

        return minstep;
}

inline float getPatchPixSize(Patch &ptch)
{
        return fmaxf(getTarCamPixSize(ptch),getRefCamPixSize(ptch));
}
*/

//inline void computeHomography(device float* out_H,
//                              device            const DeviceCameraParams& rcDeviceCamParams,
//                              device            const DeviceCameraParams& tcDeviceCamParams,
//                              device            const float3& in_p,
//                              device            const float3& in_n)
//{
//    // hartley zisserman second edition p.327 (13.2)
//    const float3 _tl = float3(0.0, 0.0, 0.0) - (rcDeviceCamParams.R * rcDeviceCamParams.C);
//    const float3 _tr = float3(0.0, 0.0, 0.0) - (tcDeviceCamParams.R * tcDeviceCamParams.C);
//
//    const float3 p = (rcDeviceCamParams.R * (in_p - rcDeviceCamParams.C));
//    float3 n = (rcDeviceCamParams.R * in_n);
//    n = normalize(n);
//    const float d = -dot(n, p);
//
////    float RrT[9];
////    M3x3transpose(RrT, rcDeviceCamParams.R);
//    float3x3 RrT = transpose(rcDeviceCamParams.R);
//
////    float tmpRr[9];
////    M3x3mulM3x3(tmpRr, tcDeviceCamParams.R, RrT);
//    float3x3 tmpRr = tcDeviceCamParams.R * RrT;
//
//    const float3 tr = _tr - (tmpRr * _tl);
//
//    float3x3 tmp;
//    float3x3 tmp1;
//    outerMultiply(tmp, tr, n / d);
//    tmp = tmpRr - tmp;
////    M3x3minusM3x3(tmp, tmpRr, tmp);
//    tmp1 = tcDeviceCamParams.K * tmp;
//    tmp = tmp1 * rcDeviceCamParams.iK;
////    M3x3mulM3x3(tmp1, tcDeviceCamParams.K, tmp);
////    M3x3mulM3x3(tmp, tmp1, rcDeviceCamParams.iK);
//
//    out_H[0] = tmp[0];
//    out_H[1] = tmp[1];
//    out_H[2] = tmp[2];
//    out_H[3] = tmp[3];
//    out_H[4] = tmp[4];
//    out_H[5] = tmp[5];
//    out_H[6] = tmp[6];
//    out_H[7] = tmp[7];
//    out_H[8] = tmp[8];
//}

/*
static float compNCCbyH(const DeviceCameraParams& rcDeviceCamParams,
                                   const DeviceCameraParams& tcDeviceCamParams,
                                   const Patch& ptch,
                                   int wsh)
{
    // get R and T image 2d coordinates from patch center 3d point
    const float2 rp = project3DPoint(rcDeviceCamParams.P, patch.p);
    //const float2 tp = project3DPoint(tcDeviceCamParams.P, patch.p);

    float H[9];
    computeHomography(H, rcDeviceCamParams, tcDeviceCamParams, ptch.p, ptch.n);

    simStat sst = simStat();

    for(int xp = -wsh; xp <= wsh; ++xp)
    {
        for(int yp = -wsh; yp <= wsh; ++yp)
        {
            float2 rpc;
            float2 tpc;
            rpc.x = rp.x + (float)xp;
            rpc.y = rp.y + (float)yp;
            tpc = V2M3x3mulV2(H, rpc);

            float2 g;
            g.x = 255.0f * tex2D(rtex, rpc.x + 0.5f, rpc.y + 0.5f);
            g.y = 255.0f * tex2D(ttex, tpc.x + 0.5f, tpc.y + 0.5f);
            sst.update(g);
        }
    }

    sst.computeSim();

    return sst.sim;
}
*/

/**
 * @brief Compute Normalized Cross-Correlation of a full square patch at given half-width.
 *
 * @tparam TInvertAndFilter invert and filter output similarity value
 *
 * @param[in] rcDeviceCameraParamsId the R camera parameters in device constant memory array
 * @param[in] tcDeviceCameraParamsId the T camera parameters in device constant memory array
 * @param[in] rcMipmapImage_tex the R camera mipmap image texture
 * @param[in] tcMipmapImage_tex the T camera mipmap image texture
 * @param[in] rcLevelWidth the R camera image width at given mipmapLevel
 * @param[in] rcLevelHeight the R camera image height at given mipmapLevel
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] tcLevelHeight the T camera image height at given mipmapLevel
 * @param[in] mipmapLevel the workflow current mipmap level (e.g. SGM=1.f, Refine=0.f)
 * @param[in] wsh the half-width of the patch
 * @param[in] invGammaC the inverted strength of grouping by color similarity
 * @param[in] invGammaP the inverted strength of grouping by proximity
 * @param[in] useConsistentScale enable consistent scale patch comparison
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] patch the input patch struct
 *
 * @return similarity value in range (-1.f, 0.f) or (0.f, 1.f) if TinvertAndFilter enabled
 *         special cases:
 *          -> infinite similarity value: 1
 *          -> invalid/uninitialized/masked similarity: INF_F
 */
//template<bool TInvertAndFilter>
inline float compNCCby3DptsYK(device const DeviceCameraParams& rcDeviceCamParams,
                              device           const DeviceCameraParams& tcDeviceCamParams,
                              texture2d<half> rcMipmapImage_tex[[texture(0)]],
                              texture2d<half> tcMipmapImage_tex[[texture(1)]],
                              device           const unsigned int& rcLevelWidth,
                              device           const unsigned int& rcLevelHeight,
                              device           const unsigned int& tcLevelWidth,
                              device           const unsigned int& tcLevelHeight,
                              device           const float& mipmapLevel,
                              device           const int& wsh,
                              device           const float& invGammaC,
                              device           const float& invGammaP,
                              device           const bool& useConsistentScale,
                              thread bool TInvertAndFilter,
                              thread           const Patch& patch)
{
    // get R and T image 2d coordinates from patch center 3d point
    const float2 rp = project3DPoint(rcDeviceCamParams.P, patch.p);
    const float2 tp = project3DPoint(tcDeviceCamParams.P, patch.p);

    // image 2d coordinates margin
    const float dd = wsh + 2.0f; // TODO: FACA

    // check R and T image 2d coordinates
    if((rp.x < dd) || (rp.x > float(rcLevelWidth  - 1) - dd) ||
       (tp.x < dd) || (tp.x > float(tcLevelWidth  - 1) - dd) ||
       (rp.y < dd) || (rp.y > float(rcLevelHeight - 1) - dd) ||
       (tp.y < dd) || (tp.y > float(tcLevelHeight - 1) - dd))
    {
        return INF_F; // uninitialized
    }

    // compute inverse width / height
    // note: useful to compute normalized coordinates
    const float rcInvLevelWidth  = 1.f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.f / float(tcLevelHeight);

    // initialize R and T mipmap image level at the given mipmap image level
    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;

    // update R and T mipmap image level in order to get consistent scale patch comparison
    if(useConsistentScale)
    {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel, mipmapLevel, rcDeviceCamParams, tcDeviceCamParams, rp, tp, patch.p);
    }

    // create and initialize SimStat struct
    simStat sst;

    // compute patch center color (CIELAB) at R and T mipmap image level
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);
//    const float4 rcCenterColor = tex2DLod<float4>(rcMipmapImage_tex, (rp.x + 0.5f) * rcInvLevelWidth, (rp.y + 0.5f) * rcInvLevelHeight, rcMipmapLevel);
//    const float4 tcCenterColor = tex2DLod<float4>(tcMipmapImage_tex, (tp.x + 0.5f) * tcInvLevelWidth, (tp.y + 0.5f) * tcInvLevelHeight, tcMipmapLevel);
    const half4 rcCenterColor = rcMipmapImage_tex.sample(textureSampler, float2((rp.x + 0.5f) * rcInvLevelWidth, (rp.y + 0.5f) * rcInvLevelHeight), level(rcMipmapLevel));
    const half4 tcCenterColor = tcMipmapImage_tex.sample(textureSampler, float2((tp.x + 0.5f) * tcInvLevelWidth, (tp.y + 0.5f) * tcInvLevelHeight), level(tcMipmapLevel));

    // check the alpha values of the patch pixel center of the R and T cameras
    if(rcCenterColor.w < ALICEVISION_DEPTHMAP_RC_MIN_ALPHA || tcCenterColor.w < ALICEVISION_DEPTHMAP_TC_MIN_ALPHA)
    {
        return INF_F; // masked
    }

    // compute patch (wsh*2+1)x(wsh*2+1)
    for(int yp = -wsh; yp <= wsh; ++yp)
    {
        for(int xp = -wsh; xp <= wsh; ++xp)
        {
            // get 3d point
            const float3 p = patch.p + patch.x * float(patch.d * float(xp)) + patch.y * float(patch.d * float(yp));

            // get R and T image 2d coordinates from 3d point
            const float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
            const float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

            // get R and T image color (CIELAB) from 2d coordinates
//            const float4 rcPatchCoordColor = tex2DLod<float4>(rcMipmapImage_tex, (rpc.x + 0.5f) * rcInvLevelWidth, (rpc.y + 0.5f) * rcInvLevelHeight, rcMipmapLevel);
//            const float4 tcPatchCoordColor = tex2DLod<float4>(tcMipmapImage_tex, (tpc.x + 0.5f) * tcInvLevelWidth, (tpc.y + 0.5f) * tcInvLevelHeight, tcMipmapLevel);
            const half4 rcPatchCoordColor = rcMipmapImage_tex.sample(textureSampler, float2((rpc.x + 0.5f) * rcInvLevelWidth, (rpc.y + 0.5f) * rcInvLevelHeight), level(rcMipmapLevel));
            const half4 tcPatchCoordColor = tcMipmapImage_tex.sample(textureSampler, float2((tpc.x + 0.5f) * tcInvLevelWidth, (tpc.y + 0.5f) * tcInvLevelHeight), level(tcMipmapLevel));

            // compute weighting based on:
            // - color difference to the center pixel of the patch:
            //    - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
            //    - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
            // - distance in image to the center pixel of the patch:
            //    - low value (close to 0) means that the pixel is close to the center of the patch
            //    - high value (close to 1) means that the pixel is far from the center of the patch
            const float w = CostYKfromLab(xp, yp, float4(rcCenterColor), float4(rcPatchCoordColor), invGammaC, invGammaP) * CostYKfromLab(xp, yp, float4(tcCenterColor), float4(tcPatchCoordColor), invGammaC, invGammaP);

            // update simStat
            sst.update(rcPatchCoordColor.x, tcPatchCoordColor.x, w);
        }
    }

    if(TInvertAndFilter)
    {
        // compute patch similarity
        const float fsim = sst.computeWSim();

        // invert and filter similarity
        // apply sigmoid see: https://www.desmos.com/calculator/skmhf1gpyf
        // best similarity value was -1, worst was 0
        // best similarity value is 1, worst is still 0
        return sigmoid(0.0f, 1.0f, 0.7f, -0.7f, fsim);
    }

    // compute output patch similarity
    return sst.computeWSim();
}

/**
 * @brief Compute Normalized Cross-Correlation of a patch with an user custom patch pattern.
 *
 * @tparam TInvertAndFilter invert and filter output similarity value
 *
 * @param[in] rcDeviceCameraParamsId the R camera parameters in device constant memory array
 * @param[in] tcDeviceCameraParamsId the T camera parameters in device constant memory array
 * @param[in] rcMipmapImage_tex the R camera mipmap image texture
 * @param[in] tcMipmapImage_tex the T camera mipmap image texture
 * @param[in] rcLevelWidth the R camera image width at given mipmapLevel
 * @param[in] rcLevelHeight the R camera image height at given mipmapLevel
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] tcLevelHeight the T camera image height at given mipmapLevel
 * @param[in] mipmapLevel the workflow current mipmap level (e.g. SGM=1.f, Refine=0.f)
 * @param[in] invGammaC the inverted strength of grouping by color similarity
 * @param[in] invGammaP the inverted strength of grouping by proximity
 * @param[in] useConsistentScale enable consistent scale patch comparison
 * @param[in] patch the input patch struct
 *
 * @return similarity value in range (-1.f, 0.f) or (0.f, 1.f) if TinvertAndFilter enabled
 *         special cases:
 *          -> infinite similarity value: 1
 *          -> invalid/uninitialized/masked similarity: INF_F
 */
//template<bool TInvertAndFilter>
inline float compNCCby3DptsYK_customPatchPattern(device const DeviceCameraParams& rcDeviceCamParams,
                                                 device            const DeviceCameraParams& tcDeviceCamParams,
                                                 texture2d<half> rcMipmapImage_tex[[texture(0)]],
                                                 texture2d<half> tcMipmapImage_tex[[texture(1)]],
                                                 device            const unsigned int &rcLevelWidth,
                                                 device            const unsigned int &rcLevelHeight,
                                                 device            const unsigned int &tcLevelWidth,
                                                 device            const unsigned int &tcLevelHeight,
                                                 device            const float &mipmapLevel,
                                                 device            const float &invGammaC,
                                                 device            const float &invGammaP,
                                                 device            const bool &useConsistentScale,
                                                 thread bool TInvertAndFilter,
                                                 thread            const Patch& patch)
{
    // get R and T image 2d coordinates from patch center 3d point
    const float2 rp = project3DPoint(rcDeviceCamParams.P, patch.p);
    const float2 tp = project3DPoint(tcDeviceCamParams.P, patch.p);

    // image 2d coordinates margin
    const float dd = 2.f; // TODO: proper wsh handling

    // check R and T image 2d coordinates
    if((rp.x < dd) || (rp.x > float(rcLevelWidth  - 1) - dd) ||
       (tp.x < dd) || (tp.x > float(tcLevelWidth  - 1) - dd) ||
       (rp.y < dd) || (rp.y > float(rcLevelHeight - 1) - dd) ||
       (tp.y < dd) || (tp.y > float(tcLevelHeight - 1) - dd))
    {
        return INF_F; // uninitialized
    }

    // compute inverse width / height
    // note: useful to compute normalized coordinates
    const float rcInvLevelWidth  = 1.f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.f / float(tcLevelHeight);

    // get patch center pixel alpha at the given mipmap image level
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);


//    const float rcAlpha = tex2DLod<float4>(rcMipmapImage_tex, (rp.x + 0.5f) * rcInvLevelWidth, (rp.y + 0.5f) * rcInvLevelHeight, mipmapLevel).w; // alpha only
//    const float tcAlpha = tex2DLod<float4>(tcMipmapImage_tex, (tp.x + 0.5f) * tcInvLevelWidth, (tp.y + 0.5f) * tcInvLevelHeight, mipmapLevel).w; // alpha only
    const half rcAlpha = rcMipmapImage_tex.sample(textureSampler, float2((rp.x + 0.5f) * rcInvLevelWidth, (rp.y + 0.5f) * rcInvLevelHeight), level(mipmapLevel)).w;
    const half tcAlpha = tcMipmapImage_tex.sample(textureSampler, float2((tp.x + 0.5f) * tcInvLevelWidth, (tp.y + 0.5f) * tcInvLevelHeight), level(mipmapLevel)).w;

    // check the alpha values of the patch pixel center of the R and T cameras
    if(rcAlpha < ALICEVISION_DEPTHMAP_RC_MIN_ALPHA || tcAlpha < ALICEVISION_DEPTHMAP_TC_MIN_ALPHA)
    {
        return INF_F; // masked
    }

    // initialize R and T mipmap image level at the given mipmap image level
    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;

    // update R and T mipmap image level in order to get consistent scale patch comparison
    if(useConsistentScale)
    {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel, mipmapLevel, rcDeviceCamParams, tcDeviceCamParams, rp, tp, patch.p);
    }

    // output similarity initialization
    float fsim = 0.f;
    float wsum = 0.f;

#if 0
    for(int s = 0; s < constantPatchPattern_d.nbSubparts; ++s)
    {
        // create and initialize patch subpart SimStat
        simStat sst;

        // get patch pattern subpart
        const DevicePatchPatternSubpart& subpart = constantPatchPattern_d.subparts[s];

        // compute patch center color (CIELAB) at subpart level resolution
        const float4 rcCenterColor = tex2DLod<float4>(rcMipmapImage_tex, (rp.x + 0.5f) * rcInvLevelWidth, (rp.y + 0.5f) * rcInvLevelHeight, rcMipmapLevel + subpart.level);
        const float4 tcCenterColor = tex2DLod<float4>(tcMipmapImage_tex, (tp.x + 0.5f) * tcInvLevelWidth, (tp.y + 0.5f) * tcInvLevelHeight, tcMipmapLevel + subpart.level);

        if(subpart.isCircle)
        {
            for(int c = 0; c < subpart.nbCoordinates; ++c)
            {
                // get patch relative coordinates
                const float2& relativeCoord = subpart.coordinates[c];

                // get 3d point from relative coordinates
                const float3 p = patch.p + patch.x * float(patch.d * relativeCoord.x) + patch.y * float(patch.d * relativeCoord.y);

                // get R and T image 2d coordinates from 3d point
                const float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
                const float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

                // get R and T image color (CIELAB) from 2d coordinates
                const float4 rcPatchCoordColor = tex2DLod<float4>(rcMipmapImage_tex, (rpc.x + 0.5f) * rcInvLevelWidth, (rpc.y + 0.5f) * rcInvLevelHeight, rcMipmapLevel + subpart.level);
                const float4 tcPatchCoordColor = tex2DLod<float4>(tcMipmapImage_tex, (tpc.x + 0.5f) * tcInvLevelWidth, (tpc.y + 0.5f) * tcInvLevelHeight, tcMipmapLevel + subpart.level);

                // compute weighting based on color difference to the center pixel of the patch:
                // - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
                // - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
                const float w = CostYKfromLab(rcCenterColor, rcPatchCoordColor, invGammaC) * CostYKfromLab(tcCenterColor, tcPatchCoordColor, invGammaC);

                // update simStat
                sst.update(rcPatchCoordColor.x, tcPatchCoordColor.x, w);
            }
        }
        else // full patch pattern
        {
            for(int yp = -subpart.wsh; yp <= subpart.wsh; ++yp)
            {
                for(int xp = -subpart.wsh; xp <= subpart.wsh; ++xp)
                {
                    // get 3d point
                    const float3 p = patch.p + patch.x * float(patch.d * float(xp) * subpart.downscale) + patch.y * float(patch.d * float(yp) * subpart.downscale);

                    // get R and T image 2d coordinates from 3d point
                    const float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
                    const float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

                    // get R and T image color (CIELAB) from 2d coordinates
                    const float4 rcPatchCoordColor = tex2DLod<float4>(rcMipmapImage_tex, (rpc.x + 0.5f) * rcInvLevelWidth, (rpc.y + 0.5f) * rcInvLevelHeight, rcMipmapLevel + subpart.level);
                    const float4 tcPatchCoordColor = tex2DLod<float4>(tcMipmapImage_tex, (tpc.x + 0.5f) * tcInvLevelWidth, (tpc.y + 0.5f) * tcInvLevelHeight, tcMipmapLevel + subpart.level);

                    // compute weighting based on:
                    // - color difference to the center pixel of the patch:
                    //    - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
                    //    - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
                    // - distance in image to the center pixel of the patch:
                    //    - low value (close to 0) means that the pixel is close to the center of the patch
                    //    - high value (close to 1) means that the pixel is far from the center of the patch
                    const float w = CostYKfromLab(xp, yp, rcCenterColor, rcPatchCoordColor, invGammaC, invGammaP) * CostYKfromLab(xp, yp, tcCenterColor, tcPatchCoordColor, invGammaC, invGammaP);

                    // update simStat
                    sst.update(rcPatchCoordColor.x, tcPatchCoordColor.x, w);
                }
            }
        }

        // compute patch subpart similarity
        const float fsimSubpart = sst.computeWSim();

        // similarity value in range (-1.f, 0.f) or invalid
        if(fsimSubpart < 0.f)
        {
            // add patch pattern subpart similarity to patch similarity
            if(TInvertAndFilter)
            {
                // invert and filter similarity
                // apply sigmoid see: https://www.desmos.com/calculator/skmhf1gpyf
                // best similarity value was -1, worst was 0
                // best similarity value is 1, worst is still 0
                const float fsimInverted = sigmoid(0.0f, 1.0f, 0.7f, -0.7f, fsimSubpart);
                fsim += fsimInverted * subpart.weight;

            }
            else
            {
                // weight and add similarity
                fsim += fsimSubpart * subpart.weight;
            }

            // sum subpart weight
            wsum += subpart.weight;
        }
    }

    // invalid patch similarity
    if(wsum == 0.f)
    {
        return INF_F;
    }

    if(TInvertAndFilter)
    {
        // for now, we do not average
        return fsim;
    }

    // output average similarity
    return (fsim / wsum);
#else
    return 0.0f;
#endif
}

} // namespace depthMap

