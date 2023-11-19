// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "volumeIO.hpp"

#include <SoftVisionLog.h>
#include <mvsData/Point3d.hpp>
#include <mvsData/Matrix3x3.hpp>
#include <mvsData/Matrix3x4.hpp>
#include <mvsData/OrientedPoint.hpp>
#include <mvsData/geometry.hpp>
#include <mvsData/jetColorMap.hpp>
#include <mvsUtils/common.hpp>
#include <mvsUtils/fileIO.hpp>
#include <sfmDataIO/sfmDataIO.hpp>
#include <depthMap/BufPtr.hpp>
#include <utils/strUtils.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>


namespace depthMap {

void exportSimilaritySamplesCSV(DeviceBuffer* in_volumeSim_hmh,
                                const std::vector<float>& in_depths,
                                const std::string& name, 
                                const SgmParams& sgmParams,
                                const std::string& filepath,
                                const ROI& roi)
{
    const ROI downscaledRoi = downscaleROI(roi, sgmParams.scale * sgmParams.stepXY);

    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];

    const int sampleSize = 3;

    const int xOffset = std::floor(downscaledRoi.width()  / (sampleSize + 1.0f));
    const int yOffset = std::floor(downscaledRoi.height() / (sampleSize + 1.0f));

    std::vector<Point2d> ptsCoords(sampleSize*sampleSize);
    std::vector<std::vector<float>> ptsDepths(sampleSize*sampleSize);

    for (int iy = 0; iy < sampleSize; ++iy)
    {
        for (int ix = 0; ix < sampleSize; ++ix)
        {
            const int ptIdx = iy * sampleSize + ix;
            const int x = (ix + 1) * xOffset;
            const int y = (iy + 1) * yOffset;

            ptsCoords.at(ptIdx) = {x, y};
            std::vector<float>& pDepths = ptsDepths.at(ptIdx);

            pDepths.reserve(in_depths.size());

            for(int iz = 0; iz < in_depths.size(); ++iz)
            {
                const float simValue = float(*get3DBufferAt_h<TSim>((TSim*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, x, y, iz));
                pDepths.push_back(simValue);
            }
        }
    }

    std::stringstream ss;

    ss << name << "\n";

    for(int i = 0; i < ptsCoords.size(); ++i)
    {
        const Point2d& coord = ptsCoords.at(i);
        ss << "p" << (i + 1) << " (x: " << coord.x << ", y: " << coord.y <<  ");";
        for(const float depth : ptsDepths.at(i))
            ss << depth << ";";
        ss << "\n";
    }

    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if(file.is_open())
        file << ss.str();
}

void exportSimilaritySamplesCSV(DeviceBuffer* in_volumeSim_hmh,
                                const std::string& name, 
                                const RefineParams& refineParams,
                                const std::string& filepath,
                                const ROI& roi)
{
    const ROI downscaledRoi = downscaleROI(roi, refineParams.scale * refineParams.stepXY);

    const size_t volDimZ = [in_volumeSim_hmh getSize].depth;
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];

    const int sampleSize = 3;

    const int xOffset = std::floor(downscaledRoi.width()  / (sampleSize + 1.0f));
    const int yOffset = std::floor(downscaledRoi.height() / (sampleSize + 1.0f));

    std::vector<Point2d> ptsCoords(sampleSize*sampleSize);
    std::vector<std::vector<float>> ptsDepths(sampleSize * sampleSize);

    for(int iy = 0; iy < sampleSize; ++iy)
    {
        for(int ix = 0; ix < sampleSize; ++ix)
        {
            const int ptIdx = iy * sampleSize + ix;
            const int x = (ix + 1) * xOffset;
            const int y = (iy + 1) * yOffset;

            ptsCoords.at(ptIdx) = {x, y};
            std::vector<float>& pDepths = ptsDepths.at(ptIdx);

            pDepths.reserve(volDimZ);

            for(int iz = 0; iz < volDimZ; ++iz)
            {
                const float simValue = float(*get3DBufferAt_h<TSimRefine>((TSimRefine*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, x, y, iz));
                pDepths.push_back(simValue);
            }
        }
    }

    std::stringstream ss;

    ss << name << "\n";

    for(int i = 0; i < ptsCoords.size(); ++i)
    {
        const Point2d& coord = ptsCoords.at(i);
        ss << "p" << (i + 1) << " (x: " << coord.x << ", y: " << coord.y <<  ");";
        for(const float depth : ptsDepths.at(i))
            ss << depth << ";";
        ss << "\n";
    }

    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if(file.is_open())
        file << ss.str();
}

void exportSimilarityVolume(DeviceBuffer* in_volumeSim_hmh,
                            const std::vector<float>& in_depths,
                            const mvsUtils::MultiViewParams& mp, 
                            int camIndex, 
                            const SgmParams& sgmParams, 
                            const std::string& filepath, 
                            const ROI& roi)
{
    sfmData::SfMData pointCloud;
    const int xyStep = 10;

    IndexT landmarkId;
    
    MTLSize volDim = [in_volumeSim_hmh getSize];
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];

    for (int vy = 0; vy < volDim.height; vy += xyStep)
    {
        for (int vx = 0; vx < volDim.width; vx += xyStep)
        {
            const double x = roi.x.begin + (vx * sgmParams.scale * sgmParams.stepXY);
            const double y = roi.y.begin + (vy * sgmParams.scale * sgmParams.stepXY);

            for(int vz = 0; vz < in_depths.size(); ++vz)
            {
                const double planeDepth = in_depths[vz];
                const Point3d planen = (mp.iRArr[camIndex] * Point3d(0.0f, 0.0f, 1.0f)).normalize();
                const Point3d planep = mp.CArr[camIndex] + planen * planeDepth;
                const Point3d v = (mp.iCamArr[camIndex] * Point2d(x,y)).normalize();
                const Point3d p = linePlaneIntersect(mp.CArr[camIndex], v, planep, planen);

                const float maxValue = 80.f;
                float simValue = *get3DBufferAt_h<TSim>((TSim*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz);
                if (simValue > maxValue)
                    continue;
                const rgb c = getRGBFromJetColorMap(simValue / maxValue);
                pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));

                ++landmarkId;
            }
        }
    }

    
    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

void exportSimilarityVolumeCross(DeviceBuffer* in_volumeSim_hmh,
                                 const std::vector<float>& in_depths,
                                 const mvsUtils::MultiViewParams& mp, 
                                 int camIndex, 
                                 const SgmParams& sgmParams,
                                 const std::string& filepath,
                                 const ROI& roi)
{
    sfmData::SfMData pointCloud;

    IndexT landmarkId;

    MTLSize volDim = [in_volumeSim_hmh getSize];
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];

    for(int vz = 0; vz < in_depths.size(); ++vz)
    {
        for(int vy = 0; vy < volDim.height; ++vy)
        {
            const bool vyCenter = (vy >= volDim.height/2) && ((vy-1)< volDim.height/2);
            const int xIdxStart = (vyCenter ? 0 : (volDim.width / 2));
            const int xIdxStop = (vyCenter ? volDim.width : (xIdxStart + 1));

            for(int vx = xIdxStart; vx < xIdxStop; ++vx)
            {
                const double x = roi.x.begin + (vx * sgmParams.scale * sgmParams.stepXY);
                const double y = roi.y.begin + (vy * sgmParams.scale * sgmParams.stepXY);
                const double planeDepth = in_depths[vz];
                const Point3d planen = (mp.iRArr[camIndex] * Point3d(0.0f, 0.0f, 1.0f)).normalize();
                const Point3d planep = mp.CArr[camIndex] + planen * planeDepth;
                const Point3d v = (mp.iCamArr[camIndex] * Point2d(x,y)).normalize();
                const Point3d p = linePlaneIntersect(mp.CArr[camIndex], v, planep, planen);

                const float maxValue = 80.f;
                float simValue = *get3DBufferAt_h<TSim>((TSim*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz);

                if(simValue > maxValue)
                    continue;

                const rgb c = getRGBFromJetColorMap(simValue / maxValue);
                pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));

                ++landmarkId;
            }
        }
    }

    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

void exportSimilarityVolumeCross(DeviceBuffer* in_volumeSim_hmh,
                                 DeviceBuffer* in_depthSimMapSgmUpscale_hmh,
                                 const mvsUtils::MultiViewParams& mp, 
                                 int camIndex, 
                                 const RefineParams& refineParams,
                                 const std::string& filepath, 
                                 const ROI& roi)
{
    sfmData::SfMData pointCloud;

    MTLSize volDim = [in_volumeSim_hmh getSize];
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];

    IndexT landmarkId = 0;

    for(int vy = 0; vy < volDim.height; ++vy)
    {
        const bool vyCenter = ((vy*2) == volDim.height);
        const int xIdxStart = (vyCenter ? 0 : (volDim.width / 2));
        const int xIdxStop = (vyCenter ? volDim.width : (xIdxStart + 1));

        for(int vx = xIdxStart; vx < xIdxStop; ++vx)
        {
            const int x = roi.x.begin + (double(vx) * refineParams.scale * refineParams.stepXY);
            const int y = roi.y.begin + (double(vy) * refineParams.scale * refineParams.stepXY);
            const Point2d pix(x, y);

            const simd_float2 depthPixSizeMap = [in_depthSimMapSgmUpscale_hmh getVec2f:vx y:vy];

            if(depthPixSizeMap.x < 0.0f) // original depth invalid or masked
                continue;

            for(int vz = 0; vz < volDim.depth; ++vz)
            {
                const float simValue = float(*get3DBufferAt_h<TSimRefine>((TSimRefine*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz));

                const float maxValue = 10.f; // sum of similarity between 0 and 1
                if(simValue > maxValue)
                    continue;

                const int relativeDepthIndexOffset = vz - refineParams.halfNbDepths;
                const double depth = depthPixSizeMap.x + (relativeDepthIndexOffset * depthPixSizeMap.y); // original depth + z based pixSize offset

                const Point3d p = mp.CArr[camIndex] + (mp.iCamArr[camIndex] * pix).normalize() * depth;

                const rgb c = getRGBFromJetColorMap(simValue / maxValue);
                pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));

                ++landmarkId;
            }
        }
    }

    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

void exportSimilarityVolumeTopographicCut(DeviceBuffer* in_volumeSim_hmh,
                                          const std::vector<float>& in_depths,
                                          const mvsUtils::MultiViewParams& mp,
                                          int camIndex,
                                          const SgmParams& sgmParams,
                                          const std::string& filepath,
                                          const ROI& roi)
{
    sfmData::SfMData pointCloud;

    MTLSize volDim = [in_volumeSim_hmh getSize];
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];
    const size_t vy = size_t(divideRoundUp(int(volDim.height), 2)); // center only

    const Point3d planen = (mp.iRArr[camIndex] * Point3d(0.0f, 0.0f, 1.0f)).normalize();

    // compute min and max similarity values
    float minSim = std::numeric_limits<float>::max();
    float maxSim = std::numeric_limits<float>::min();

    for(size_t vx = 0; vx < volDim.width; ++vx)
    {
        for(size_t vz = 0; vz < volDim.depth; ++vz)
        {
            const float simValue = float(*get3DBufferAt_h<TSim>((TSim*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz));

            if(simValue > 254.f) // invalid similarity
              continue;

            maxSim = std::max(maxSim, simValue);
            minSim = std::min(minSim, simValue);
        }
    }

    const float simNorm = (maxSim == minSim) ? 0.f : (1.f / (maxSim - minSim));

    // compute each point color and position
    IndexT landmarkId = 0;

    for(size_t vx = 0; vx < volDim.width; ++vx)
    {
        const double x = roi.x.begin + (vx * sgmParams.scale * sgmParams.stepXY);
        const double y = roi.y.begin + (vy * sgmParams.scale * sgmParams.stepXY);
        const Point2d pix(x, y);

        for(size_t vz = 0; vz < in_depths.size(); ++vz)
        {
            const float simValue = float(*get3DBufferAt_h<TSim>((TSim*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz));

            if(simValue > 254.f) // invalid similarity
              continue;

            const float simValueNorm = (simValue - minSim) * simNorm;

            const double planeDepth = in_depths[vz];
            const Point3d planep = mp.CArr[camIndex] + planen * planeDepth;
            const Point3d v = (mp.iCamArr[camIndex] * (pix + Point2d(0.0, simValueNorm * 15.0))).normalize();
            const Point3d p = linePlaneIntersect(mp.CArr[camIndex], v, planep, planen);

            const rgb c = getRGBFromJetColorMap(simValueNorm);
            pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));

            ++landmarkId;
        }
    }

    // write point cloud
    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

void exportSimilarityVolumeTopographicCut(DeviceBuffer* in_volumeSim_hmh,
                                          DeviceBuffer* in_depthSimMapSgmUpscale_hmh,
                                          const mvsUtils::MultiViewParams& mp,
                                          int camIndex,
                                          const RefineParams& refineParams,
                                          const std::string& filepath,
                                          const ROI& roi)
{
    sfmData::SfMData pointCloud;

    MTLSize volDim = [in_volumeSim_hmh getSize];
    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];
    const size_t vy = size_t(divideRoundUp(int(volDim.height), 2)); // center only

    // compute min and max similarity values
    const float minSim = 0.f;
    float maxSim = std::numeric_limits<float>::epsilon();

    for(size_t vx = 0; vx < volDim.width; ++vx)
    {
        for(size_t vz = 0; vz < volDim.depth; ++vz)
        {
            const float simValue = float(*get3DBufferAt_h<TSimRefine>((TSimRefine*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz));
            maxSim = std::max(maxSim, simValue);
        }
    }

    // compute each point color and position
    IndexT landmarkId = 0;

    for(size_t vx = 0; vx < volDim.width; ++vx)
    {
        const double x = roi.x.begin + (vx * refineParams.scale * refineParams.stepXY);
        const double y = roi.y.begin + (vy * refineParams.scale * refineParams.stepXY);
        const Point2d pix(x, y);

        const simd_float2 depthPixSizeMap = [in_depthSimMapSgmUpscale_hmh getVec2f:vx y:vy];

        if(depthPixSizeMap.x < 0.0f) // middle depth (SGM) invalid or masked
            continue;

        for(size_t vz = 0; vz < volDim.depth; ++vz)
        {
            const float simValue = float(*get3DBufferAt_h<TSimRefine>((TSimRefine*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz));
            const float simValueNorm = (simValue - minSim) / (maxSim - minSim);
            const float simValueColor = 1 - simValueNorm; // best similarity value is 0, worst value is 1

            const int relativeDepthIndexOffset = vz - refineParams.halfNbDepths;
            const double depth = depthPixSizeMap.x + (relativeDepthIndexOffset * depthPixSizeMap.y); // original depth + z based pixSize offset

            const Point3d p = mp.CArr[camIndex] + (mp.iCamArr[camIndex] * (pix + Point2d(0.0, -simValueNorm * 15.0))).normalize() * depth;

            const rgb c = getRGBFromJetColorMap(simValueColor);
            pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));

            ++landmarkId;
        }
    }

    // write point cloud
    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

inline unsigned char float_to_uchar(float v)
{
    float vv = std::max(0.f, v);
    vv = std::min(255.f, vv);
    unsigned char out = vv;
    return out;
}

//inline rgb float4_to_rgb(const float4& v)
//{
//    return { float_to_uchar(v.x), float_to_uchar(v.y), float_to_uchar(v.z) };
//}

void exportColorVolume(DeviceBuffer* in_volumeSim_hmh,
                       const std::vector<float>& in_depths,
                       int startDepth, 
                       int nbDepths, 
                       const mvsUtils::MultiViewParams& mp, 
                       int camIndex, 
                       int scale, 
                       int step, 
                       const std::string& filepath,
                       const ROI& roi)
{
    LOG_INFO("TODO: exportColorVolume IMPL...");
//    sfmData::SfMData pointCloud;
//    int xyStep = 10;
//
//    IndexT landmarkId;
//
//    MTLSize volDim = [in_volumeSim_hmh getSize];
//    const size_t spitch = [in_volumeSim_hmh getBytesUpToDim:1];
//    const size_t pitch = [in_volumeSim_hmh getBytesUpToDim:0];
//
//    LOG_X("DepthMap exportColorVolume: " << volDim.width << " x " << volDim.height << " x " << nbDepths << ", volDim.depth=" << volDim.depth << ", xyStep=" << xyStep << ".");
//
//
//    for (int vy = 0; vy < volDim.height; vy += xyStep)
//    {
//        for (int vx = 0; vx < volDim.width; vx += xyStep)
//        {
//            const double x = roi.x.begin + (vx * scale * step);
//            const double y = roi.y.begin + (vy * scale * step);
//
//            for(int vz = 0; vz < nbDepths; ++vz)
//            {
//                const double planeDepth = in_depths[startDepth + vz];
//                const Point3d planen = (mp.iRArr[camIndex] * Point3d(0.0f, 0.0f, 1.0f)).normalize();
//                const Point3d planep = mp.CArr[camIndex] + planen * planeDepth;
//                const Point3d v = (mp.iCamArr[camIndex] * Point2d(x, y)).normalize();
//                const Point3d p = linePlaneIntersect(mp.CArr[camIndex], v, planep, planen);
//
//                vector_float4 colorValue = *get3DBufferAt_h<vector_float4>((vector_float4*)[in_volumeSim_hmh getBufferPtr], spitch, pitch, vx, vy, vz);
//                const rgb c = float4_to_rgb(colorValue); // TODO: convert Lab color into sRGB color
//                pointCloud.getLandmarks()[landmarkId] = sfmData::Landmark(Vec3(p.x, p.y, p.z), feature::EImageDescriberType::UNKNOWN, sfmData::Observations(), image::RGBColor(c.r, c.g, c.b));
//
//                ++landmarkId;
//            }
//        }
//    }
//
//    sfmDataIO::Save(pointCloud,utils::GetParentPath(filepath), utils::GetFileName(filepath), sfmDataIO::ESfMData::STRUCTURE);
}

} // namespace depthMap
