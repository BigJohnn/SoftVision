// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "jsonIO.hpp"
//#include <camera/camera.hpp>
//#include <sfmDataIO/viewIO.hpp>

//#include <boost/property_tree/json_parser.hpp>

#include <memory>
#include <cassert>

#include <cJSON/cJSON.h>
#include <SoftVisionLog.h>

#include <camera/IntrinsicsScaleOffsetDisto.hpp>
#include <camera/PinholeRadial.hpp>
#include <utils/PngUtils.h>

namespace sfmDataIO {

//void saveView(const std::string& name, const sfmData::View& view, bpt::ptree& parentTree)
//{
//  bpt::ptree viewTree;
//
//  if(view.getViewId() != UndefinedIndexT)
//    viewTree.put("viewId", view.getViewId());
//
//  if(view.getPoseId() != UndefinedIndexT)
//    viewTree.put("poseId", view.getPoseId());
//
//  if(view.isPartOfRig())
//  {
//    viewTree.put("rigId", view.getRigId());
//    viewTree.put("subPoseId", view.getSubPoseId());
//  }
//
//  if(view.getFrameId() != UndefinedIndexT)
//    viewTree.put("frameId", view.getFrameId());
//
//  if(view.getIntrinsicId() != UndefinedIndexT)
//    viewTree.put("intrinsicId", view.getIntrinsicId());
//
//  if(view.getResectionId() != UndefinedIndexT)
//    viewTree.put("resectionId", view.getResectionId());
//
//  if(view.isPoseIndependant() == false)
//    viewTree.put("isPoseIndependant", view.isPoseIndependant());
//
//  viewTree.put("path", view.getImagePath());
//  viewTree.put("width", view.getWidth());
//  viewTree.put("height", view.getHeight());
//
//  // metadata
//  {
//    bpt::ptree metadataTree;
//
//    for(const auto& metadataPair : view.getMetadata())
//      metadataTree.put(metadataPair.first, metadataPair.second);
//
//    viewTree.add_child("metadata", metadataTree);
//  }
//
//  // ancestors
//  if (!view.getAncestors().empty())
//  {
//    bpt::ptree ancestorsTree;
//
//    for(const auto & ancestor : view.getAncestors())
//    {
//      bpt::ptree ancestorTree;
//      ancestorTree.put("", ancestor);
//      ancestorsTree.push_back(std::make_pair("", ancestorTree));
//    }
//
//    viewTree.add_child("ancestors", ancestorsTree);
//  }
//
//  parentTree.push_back(std::make_pair(name, viewTree));
//}
//
//void loadView(sfmData::View& view, bpt::ptree& viewTree)
//{
//  view.setViewId(viewTree.get<IndexT>("viewId", UndefinedIndexT));
//  view.setPoseId(viewTree.get<IndexT>("poseId", UndefinedIndexT));
//
//  if(viewTree.count("rigId"))
//  {
//    view.setRigAndSubPoseId(
//      viewTree.get<IndexT>("rigId"),
//      viewTree.get<IndexT>("subPoseId"));
//  }
//
//  view.setFrameId(viewTree.get<IndexT>("frameId", UndefinedIndexT));
//  view.setIntrinsicId(viewTree.get<IndexT>("intrinsicId", UndefinedIndexT));
//  view.setResectionId(viewTree.get<IndexT>("resectionId", UndefinedIndexT));
//  view.setIndependantPose(viewTree.get<bool>("isPoseIndependant", true));
//
//  view.setImagePath(viewTree.get<std::string>("path"));
//  view.setWidth(viewTree.get<std::size_t>("width", 0));
//  view.setHeight(viewTree.get<std::size_t>("height", 0));
//
//  if (viewTree.count("ancestors"))
//  {
//    for (bpt::ptree::value_type &ancestor : viewTree.get_child("ancestors"))
//    {
//      view.addAncestor(ancestor.second.get_value<IndexT>());
//    }
//  }
//
//  // metadata
//  if(viewTree.count("metadata"))
//    for(bpt::ptree::value_type &metaDataNode : viewTree.get_child("metadata"))
//      view.addMetadata(metaDataNode.first, metaDataNode.second.data());
//
//}
//
//void saveIntrinsic(const std::string& name, IndexT intrinsicId, const std::shared_ptr<camera::IntrinsicBase>& intrinsic, bpt::ptree& parentTree)
//{
//  bpt::ptree intrinsicTree;
//
//  camera::EINTRINSIC intrinsicType = intrinsic->getType();
//
//  intrinsicTree.put("intrinsicId", intrinsicId);
//  intrinsicTree.put("width", intrinsic->w());
//  intrinsicTree.put("height", intrinsic->h());
//  intrinsicTree.put("sensorWidth", intrinsic->sensorWidth());
//  intrinsicTree.put("sensorHeight", intrinsic->sensorHeight());
//  intrinsicTree.put("serialNumber", intrinsic->serialNumber());
//  intrinsicTree.put("type", camera::EINTRINSIC_enumToString(intrinsicType));
//  intrinsicTree.put("initializationMode", camera::EInitMode_enumToString(intrinsic->getInitializationMode()));
//
//  std::shared_ptr<camera::IntrinsicsScaleOffset> intrinsicScaleOffset = std::dynamic_pointer_cast<camera::IntrinsicsScaleOffset>(intrinsic);
//  if (intrinsicScaleOffset)
//  {
//    
//    const double initialFocalLengthMM = (intrinsicScaleOffset->getInitialScale().x() > 0) ? intrinsicScaleOffset->sensorWidth() * intrinsicScaleOffset->getInitialScale().x() / double(intrinsic->w()): -1;
//    const double focalLengthMM = intrinsicScaleOffset->sensorWidth() * intrinsicScaleOffset->getScale().x() / double(intrinsic->w());
//    const double focalRatio = intrinsicScaleOffset->getScale().x() / intrinsicScaleOffset->getScale().y();
//    const double pixelAspectRatio = 1.0 / focalRatio;
//
//    intrinsicTree.put("initialFocalLength", initialFocalLengthMM);
//    intrinsicTree.put("focalLength", focalLengthMM);
//    intrinsicTree.put("pixelRatio", pixelAspectRatio);
//    intrinsicTree.put("pixelRatioLocked", intrinsicScaleOffset->isRatioLocked());
//
//    saveMatrix("principalPoint", intrinsicScaleOffset->getOffset(), intrinsicTree);
//  }
//
//  std::shared_ptr<camera::IntrinsicsScaleOffsetDisto> intrinsicScaleOffsetDisto = std::dynamic_pointer_cast<camera::IntrinsicsScaleOffsetDisto>(intrinsic);
//  if (intrinsicScaleOffsetDisto)
//  {
//    bpt::ptree distParamsTree;
//
//    std::shared_ptr<camera::Distortion> distortionObject = intrinsicScaleOffsetDisto->getDistortion();
//    if (distortionObject)
//    {
//        for (double param : distortionObject->getParameters())
//        {
//            bpt::ptree paramTree;
//            paramTree.put("", param);
//            distParamsTree.push_back(std::make_pair("", paramTree));
//        }
//    }
//    intrinsicTree.put("distortionInitializationMode", camera::EInitMode_enumToString(intrinsicScaleOffsetDisto->getDistortionInitializationMode()));
//
//    intrinsicTree.add_child("distortionParams", distParamsTree);
//
//    bpt::ptree undistParamsTree;
//
//    std::shared_ptr<camera::Undistortion> undistortionObject = intrinsicScaleOffsetDisto->getUndistortion();
//    if (undistortionObject)
//    {
//        saveMatrix("undistortionOffset", undistortionObject->getOffset(), intrinsicTree);
//
//        for (double param : undistortionObject->getParameters())
//        {
//            bpt::ptree paramTree;
//            paramTree.put("", param);
//            undistParamsTree.push_back(std::make_pair("", paramTree));
//        }
//    }
//    else
//    {
//        saveMatrix("undistortionOffset", Vec2{0.0, 0.0}, intrinsicTree);
//    }
//
//    intrinsicTree.add_child("undistortionParams", undistParamsTree);
//  }
//
//  std::shared_ptr<camera::EquiDistant> intrinsicEquidistant = std::dynamic_pointer_cast<camera::EquiDistant>(intrinsic);
//  if (intrinsicEquidistant)
//  {
//    intrinsicTree.put("fisheyeCircleCenterX", intrinsicEquidistant->getCircleCenterX());
//    intrinsicTree.put("fisheyeCircleCenterY", intrinsicEquidistant->getCircleCenterY());
//    intrinsicTree.put("fisheyeCircleRadius", intrinsicEquidistant->getCircleRadius());
//  }
//
//  intrinsicTree.put("locked", intrinsic->isLocked());
//
//  parentTree.push_back(std::make_pair(name, intrinsicTree));
// 
//}
//
//void loadIntrinsic(const Version & version, IndexT& intrinsicId, std::shared_ptr<camera::IntrinsicBase>& intrinsic,
//                   bpt::ptree& intrinsicTree)
//{
//  intrinsicId = intrinsicTree.get<IndexT>("intrinsicId");
//  const unsigned int width = intrinsicTree.get<unsigned int>("width");
//  const unsigned int height = intrinsicTree.get<unsigned int>("height");
//  const double sensorWidth = intrinsicTree.get<double>("sensorWidth", 36.0);
//  const double sensorHeight = intrinsicTree.get<double>("sensorHeight", 24.0);
//  const camera::EINTRINSIC intrinsicType = camera::EINTRINSIC_stringToEnum(intrinsicTree.get<std::string>("type"));
//  const camera::EInitMode initializationMode = camera::EInitMode_stringToEnum(intrinsicTree.get<std::string>("initializationMode", camera::EInitMode_enumToString(camera::EInitMode::CALIBRATED)));
//
//  // principal point
//  Vec2 principalPoint;
//  loadMatrix("principalPoint", principalPoint, intrinsicTree);
//
//  if (version < Version(1,2,1))
//  {
//    principalPoint[0] -= (double(width) / 2.0);
//    principalPoint[1] -= (double(height) / 2.0);
//  }
//
//  // Focal length
//  Vec2 pxFocalLength;
//  if (version < Version(1,2,0))
//  {
//      pxFocalLength(0) = intrinsicTree.get<double>("pxFocalLength", -1);
//      // Only one focal value for X and Y in previous versions
//      pxFocalLength(1) = pxFocalLength(0);
//  }
//  else if (version < Version(1,2,2)) // version >= 1.2
//  {
//    loadMatrix("pxFocalLength", pxFocalLength, intrinsicTree);
//  }
//  else if (version < Version(1,2,5))
//  {
//    const double fmm = intrinsicTree.get<double>("focalLength", 1.0);
//    // pixelRatio field was actually storing the focalRatio before version 1.2.5
//    const double focalRatio = intrinsicTree.get<double>("pixelRatio", 1.0);
//
//    const double fx = (fmm / sensorWidth) * double(width);
//    const double fy = fx / focalRatio;
//
//    pxFocalLength(0) = fx;
//    pxFocalLength(1) = fy;
//  }
//  else
//  {
//    const double fmm = intrinsicTree.get<double>("focalLength", 1.0);
//    const double pixelAspectRatio = intrinsicTree.get<double>("pixelRatio", 1.0);
//
//    const double focalRatio = 1.0 / pixelAspectRatio;
//    const double fx = (fmm / sensorWidth) * double(width);
//    const double fy = fx / focalRatio;
//
//    pxFocalLength(0) = fx;
//    pxFocalLength(1) = fy;
//  }
//
//  // pinhole parameters
//  intrinsic = camera::createIntrinsic(intrinsicType, width, height, pxFocalLength(0), pxFocalLength(1), principalPoint(0), principalPoint(1));  
//  
//  intrinsic->setSerialNumber(intrinsicTree.get<std::string>("serialNumber"));
//  intrinsic->setInitializationMode(initializationMode);
//  intrinsic->setSensorWidth(sensorWidth);
//  intrinsic->setSensorHeight(sensorHeight);
//
//  // intrinsic lock
//  if(intrinsicTree.get<bool>("locked", false)) {
//    intrinsic->lock();
//  }
//  else {
//    intrinsic->unlock();
//  }
//
//  std::shared_ptr<camera::IntrinsicsScaleOffset> intrinsicWithScale = std::dynamic_pointer_cast<camera::IntrinsicsScaleOffset>(intrinsic);
//  if (intrinsicWithScale != nullptr) {
//
//    if (version < Version(1, 2, 2))
//    {
//      Vec2 initialFocalLengthPx;
//      initialFocalLengthPx(0) = intrinsicTree.get<double>("pxInitialFocalLength");
//      initialFocalLengthPx(1) = (initialFocalLengthPx(0) > 0)?initialFocalLengthPx(0) * pxFocalLength(1) / pxFocalLength(0):-1;
//      intrinsicWithScale->setInitialScale(initialFocalLengthPx);
//    }
//    else 
//    {
//      double initialFocalLengthMM = intrinsicTree.get<double>("initialFocalLength");
//      
//      Vec2 initialFocalLengthPx;
//      initialFocalLengthPx(0) = (initialFocalLengthMM / sensorWidth) * double(width);
//      initialFocalLengthPx(1) = (initialFocalLengthPx(0) > 0)?initialFocalLengthPx(0) * pxFocalLength(1) / pxFocalLength(0):-1;
//
//      intrinsicWithScale->setInitialScale(initialFocalLengthPx);
//      intrinsicWithScale->setRatioLocked(intrinsicTree.get<bool>("pixelRatioLocked"));
//    }
//  }
//
//  // Load distortion
//  std::shared_ptr<camera::IntrinsicsScaleOffsetDisto> intrinsicWithDistoEnabled = std::dynamic_pointer_cast<camera::IntrinsicsScaleOffsetDisto>(intrinsic);
//  if (intrinsicWithDistoEnabled != nullptr)
//  {
//    const camera::EInitMode distortionInitializationMode =
//        camera::EInitMode_stringToEnum(
//            intrinsicTree.get<std::string>("distortionInitializationMode", camera::EInitMode_enumToString(camera::EInitMode::NONE)));
//
//    intrinsicWithDistoEnabled->setDistortionInitializationMode(distortionInitializationMode);
//
//    std::shared_ptr<camera::Distortion> distortionObject = intrinsicWithDistoEnabled->getDistortion();
//    if (distortionObject)
//    {
//        std::vector<double> distortionParams;
//        for (bpt::ptree::value_type &paramNode : intrinsicTree.get_child("distortionParams"))
//        {
//            distortionParams.emplace_back(paramNode.second.get_value<double>());
//        }
//
//        // ensure that we have the right number of params
//        if (distortionParams.size() == distortionObject->getParameters().size())
//        {
//            distortionObject->setParameters(distortionParams);
//        }
//        else
//        {
//            intrinsicWithDistoEnabled->setDistortionObject(nullptr);
//        }
//    }
//
//    std::shared_ptr<camera::Undistortion> undistortionObject = intrinsicWithDistoEnabled->getUndistortion();
//    if (undistortionObject)
//    {
//        std::vector<double> undistortionParams;
//        for (bpt::ptree::value_type &paramNode : intrinsicTree.get_child("undistortionParams"))
//        {
//            undistortionParams.emplace_back(paramNode.second.get_value<double>());
//        }
//
//        // ensure that we have the right number of params
//        if (undistortionParams.size() == undistortionObject->getParameters().size())
//        {
//            undistortionObject->setParameters(undistortionParams);
//            Vec2 offset;
//            loadMatrix("undistortionOffset", offset, intrinsicTree);
//            undistortionObject->setOffset(offset);
//        }
//        else
//        {
//            intrinsicWithDistoEnabled->setUndistortionObject(nullptr);
//        }
//    }
//  }
//
//  // Load EquiDistant params
//  std::shared_ptr<camera::EquiDistant> intrinsicEquiDistant = std::dynamic_pointer_cast<camera::EquiDistant>(intrinsic);
//  if (intrinsicEquiDistant != nullptr)
//  {
//    intrinsicEquiDistant->setCircleCenterX(intrinsicTree.get<double>("fisheyeCircleCenterX", 0.0));
//    intrinsicEquiDistant->setCircleCenterY(intrinsicTree.get<double>("fisheyeCircleCenterY", 0.0));
//    intrinsicEquiDistant->setCircleRadius(intrinsicTree.get<double>("fisheyeCircleRadius", 1.0));
//  }
//}
//
//void saveRig(const std::string& name, IndexT rigId, const sfmData::Rig& rig, bpt::ptree& parentTree)
//{
//  bpt::ptree rigTree;
//
//  rigTree.put("rigId", rigId);
//
//  bpt::ptree rigSubPosesTree;
//
//  for(const auto& rigSubPose : rig.getSubPoses())
//  {
//    bpt::ptree rigSubPoseTree;
//
//    rigSubPoseTree.put("status", sfmData::ERigSubPoseStatus_enumToString(rigSubPose.status));
//    savePose3("pose", rigSubPose.pose, rigSubPoseTree);
//
//    rigSubPosesTree.push_back(std::make_pair("", rigSubPoseTree));
//  }
//
//  rigTree.add_child("subPoses", rigSubPosesTree);
//
//  parentTree.push_back(std::make_pair(name, rigTree));
//}
//
//
//void loadRig(IndexT& rigId, sfmData::Rig& rig, bpt::ptree& rigTree)
//{
//  rigId =  rigTree.get<IndexT>("rigId");
//  rig = sfmData::Rig(rigTree.get_child("subPoses").size());
//  int subPoseId = 0;
//
//  for(bpt::ptree::value_type& subPoseNode : rigTree.get_child("subPoses"))
//  {
//    bpt::ptree& subPoseTree = subPoseNode.second;
//
//    sfmData::RigSubPose subPose;
//
//    subPose.status = sfmData::ERigSubPoseStatus_stringToEnum(subPoseTree.get<std::string>("status"));
//    loadPose3("pose", subPose.pose, subPoseTree);
//
//    rig.setSubPose(subPoseId++, subPose);
//  }
//}
//
//void saveLandmark(const std::string& name, IndexT landmarkId, const sfmData::Landmark& landmark, bpt::ptree& parentTree, bool saveObservations, bool saveFeatures)
//{
//  bpt::ptree landmarkTree;
//
//  landmarkTree.put("landmarkId", landmarkId);
//  landmarkTree.put("descType", feature::EImageDescriberType_enumToString(landmark.descType));
//
//  saveMatrix("color", landmark.rgb, landmarkTree);
//  saveMatrix("X", landmark.X, landmarkTree);
//
//  // observations
//  if(saveObservations)
//  {
//    bpt::ptree observationsTree;
//    for(const auto& obsPair : landmark.observations)
//    {
//      bpt::ptree obsTree;
//
//      const sfmData::Observation& observation = obsPair.second;
//
//      obsTree.put("observationId", obsPair.first);
//
//      // features
//      if(saveFeatures)
//      {
//        obsTree.put("featureId", observation.id_feat);
//        saveMatrix("x", observation.x, obsTree);
//        obsTree.put("scale", observation.scale);
//      }
//
//      observationsTree.push_back(std::make_pair("", obsTree));
//    }
//
//    landmarkTree.add_child("observations", observationsTree);
//  }
//
//  parentTree.push_back(std::make_pair(name, landmarkTree));
//}
//
//void loadLandmark(IndexT& landmarkId, sfmData::Landmark& landmark, bpt::ptree& landmarkTree, bool loadObservations, bool loadFeatures)
//{
//  landmarkId = landmarkTree.get<IndexT>("landmarkId");
//  landmark.descType = feature::EImageDescriberType_stringToEnum(landmarkTree.get<std::string>("descType"));
//
//  loadMatrix("color", landmark.rgb, landmarkTree);
//  loadMatrix("X", landmark.X, landmarkTree);
//
//  // observations
//  if(loadObservations)
//  {
//    for(bpt::ptree::value_type &obsNode : landmarkTree.get_child("observations"))
//    {
//      bpt::ptree& obsTree = obsNode.second;
//
//      sfmData::Observation observation;
//
//      if(loadFeatures)
//      {
//        observation.id_feat = obsTree.get<IndexT>("featureId");
//        loadMatrix("x", observation.x, obsTree);
//        observation.scale = obsTree.get<double>("scale", 0.0);
//      }
//
//      landmark.observations.emplace(obsTree.get<IndexT>("observationId"), observation);
//    }
//  }
//}


bool saveJSON(const sfmData::SfMData& sfmData, const std::string& foldername, const std::string& filename, ESfMData partFlag)
{
//  const Vec3i version = {ALICEVISION_SFMDATAIO_VERSION_MAJOR, ALICEVISION_SFMDATAIO_VERSION_MINOR, ALICEVISION_SFMDATAIO_VERSION_REVISION};

  // save flags
  const bool saveViews = (partFlag & VIEWS) == VIEWS;
  const bool saveIntrinsics = (partFlag & INTRINSICS) == INTRINSICS;
//  const bool saveExtrinsics = (partFlag & EXTRINSICS) == EXTRINSICS;
//  const bool saveStructure = (partFlag & STRUCTURE) == STRUCTURE;
//  const bool saveControlPoints = (partFlag & CONTROL_POINTS) == CONTROL_POINTS;
//  const bool saveFeatures = (partFlag & OBSERVATIONS_WITH_FEATURES) == OBSERVATIONS_WITH_FEATURES;
//  const bool saveObservations = saveFeatures || ((partFlag & OBSERVATIONS) == OBSERVATIONS);

    bool ret = true;
    
    cJSON *monitor = cJSON_CreateObject();
    if(!monitor) {
        LOG_ERROR("Failed in creating json object 'root'.");
        return false;
    }
    
  // views
  if(saveViews && !sfmData.getViews().empty())
  {
      cJSON *viewObjs = cJSON_CreateArray();
      if (!viewObjs)
      {
          LOG_ERROR("Failed in creating json object 'views'.");
          return false;
      }
      cJSON_AddItemToObject(monitor, "views", viewObjs);
      
      for(auto&& view : sfmData.views)
      {
          auto* viewObj = cJSON_CreateObject();
          if(!viewObj) {
              LOG_ERROR("Failed in creating json object 'view'.");
              return false;
          }
          if(!cJSON_AddStringToObject(viewObj, "path", (foldername + std::to_string(view.second->getViewId()) + ".png").c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(viewObj, "viewId", std::to_string(view.second->getViewId()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(viewObj, "poseId", std::to_string(view.second->getPoseId()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(viewObj, "intrinsicId", std::to_string(view.second->getIntrinsicId()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(viewObj, "width", std::to_string(view.second->getWidth()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(viewObj, "height", std::to_string(view.second->getHeight()).c_str())) {
              ret = false;
          }
          
          cJSON_AddItemToArray(viewObjs, viewObj);
      }
  }

  // intrinsics
  if(saveIntrinsics && !sfmData.getIntrinsics().empty())
  {
      cJSON *intrinsicObjs = cJSON_CreateArray();
      if (!intrinsicObjs)
      {
          LOG_ERROR("Failed in creating json object 'intrinsics'.");
          return false;
      }
      cJSON_AddItemToObject(monitor, "intrinsics", intrinsicObjs);
      
      for(auto&& intrinsic_map : sfmData.intrinsics)
      {
          auto* intrinsicObj = cJSON_CreateObject();
          if(!intrinsicObj) {
              LOG_ERROR("Failed in creating json object 'intrinsic'.");
              return false;
          }
          
          if(!cJSON_AddStringToObject(intrinsicObj, "intrinsicId", std::to_string(intrinsic_map.first).c_str())) {
              ret = false;
          }
          auto&& intrinsic = intrinsic_map.second;
          if(!cJSON_AddStringToObject(intrinsicObj, "width", std::to_string(intrinsic->w()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "height", std::to_string(intrinsic->h()).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "sensorWidth", std::to_string(int(intrinsic->sensorWidth())).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "sensorHeight", std::to_string(int(intrinsic->sensorHeight())).c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "type", intrinsic->getTypeStr().c_str())) {
              ret = false;
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "initializationMode", EInitMode_enumToString(intrinsic->getInitializationMode()).c_str())) {
              ret = false;
          }
          
          std::shared_ptr<camera::IntrinsicsScaleOffset> intrinsicScaleOffset =  std::dynamic_pointer_cast<camera::IntrinsicsScaleOffset>(intrinsic);
          if(intrinsicScaleOffset) {
              const double initialFocalLengthMM = (intrinsicScaleOffset->getInitialScale().x() > 0) ? intrinsicScaleOffset->sensorWidth() * intrinsicScaleOffset->getInitialScale().x() / double(intrinsic->w()): -1;
              const double focalLengthMM = intrinsicScaleOffset->sensorWidth() * intrinsicScaleOffset->getScale().x() / double(intrinsic->w());
              const double focalRatio = intrinsicScaleOffset->getScale().x() / intrinsicScaleOffset->getScale().y();
              const double pixelAspectRatio = 1.0 / focalRatio;
              
              if(!cJSON_AddStringToObject(intrinsicObj, "initialFocalLength", std::to_string(int(initialFocalLengthMM)).c_str())) {
                  ret = false;
              }
              if(!cJSON_AddStringToObject(intrinsicObj, "focalLength", std::to_string(focalLengthMM).c_str())) {
                  ret = false;
              }
              if(!cJSON_AddStringToObject(intrinsicObj, "pixelRatio", std::to_string(pixelAspectRatio).c_str())) {
                  ret = false;
              }
              if(!cJSON_AddStringToObject(intrinsicObj, "pixelRatioLocked", std::to_string(intrinsicScaleOffset->isRatioLocked()).c_str())) {
                  ret = false;
              }
              
              auto&& principalPoint = intrinsicScaleOffset->getOffset();
              
              const char* ppVals[2] = {std::to_string(principalPoint.x()).c_str(),
                  std::to_string(principalPoint.y()).c_str()};
              cJSON *principalPointObj = cJSON_CreateStringArray(ppVals, 2);
              if (!principalPointObj)
              {
                  LOG_ERROR("Failed in creating json object 'principalPointObj'.");
                  return false;
              }
              
              if(!cJSON_AddStringToObject(intrinsicObj, "distortionInitializationMode", EInitMode_enumToString(intrinsicScaleOffset->getDistortionInitializationMode()).c_str())) {
                  ret = false;
              }
              //TODO: distortion params
              
              cJSON_AddItemToObject(intrinsicObj, "principalPoint", principalPointObj);
          }
          if(!cJSON_AddStringToObject(intrinsicObj, "locked", std::to_string(intrinsic->isLocked()).c_str())) {
              ret = false;
          }
          
          cJSON_AddItemToArray(intrinsicObjs, intrinsicObj);
      }
      
  }
    
  FILE * fp;
  
  fp = fopen ((foldername + filename).c_str(), "w");
    
  char* sjson = cJSON_Print(monitor);
  if (sjson == NULL)
  {
    LOG_ERROR("Failed to print monitor.\n");
  }
  fprintf(fp, "%s", sjson);
  
  fclose(fp);
  cJSON_Delete(monitor);

//  //extrinsics
//  if(saveExtrinsics)
//  {
//    // poses
//    if(!sfmData.getPoses().empty())
//    {
//      bpt::ptree posesTree;
//
//      for(const auto& posePair : sfmData.getPoses())
//      {
//        bpt::ptree poseTree;
//
//        poseTree.put("poseId", posePair.first);
//        saveCameraPose("pose", posePair.second, poseTree);
//        posesTree.push_back(std::make_pair("", poseTree));
//      }
//
//      fileTree.add_child("poses", posesTree);
//    }
//
//    // rigs
//    if(!sfmData.getRigs().empty())
//    {
//      bpt::ptree rigsTree;
//
//      for(const auto& rigPair : sfmData.getRigs())
//        saveRig("", rigPair.first, rigPair.second, rigsTree);
//
//      fileTree.add_child("rigs", rigsTree);
//    }
//  }
//
//  // structure
//  if(saveStructure && !sfmData.getLandmarks().empty())
//  {
//    bpt::ptree structureTree;
//
//    for(const auto& structurePair : sfmData.getLandmarks())
//      saveLandmark("", structurePair.first, structurePair.second, structureTree, saveObservations, saveFeatures);
//
//    fileTree.add_child("structure", structureTree);
//  }
//
//  // control points
//  if(saveControlPoints && !sfmData.getControlPoints().empty())
//  {
//    bpt::ptree controlPointTree;
//
//    for(const auto& controlPointPair : sfmData.getControlPoints())
//      saveLandmark("", controlPointPair.first, controlPointPair.second, controlPointTree);
//
//    fileTree.add_child("controlPoints", controlPointTree);
//  }
//
//  // write the json file with the tree
//
//  bpt::write_json(filename, fileTree);

  return ret;
}

bool loadJSON(sfmData::SfMData& sfmData, const std::string& foldername, const std::string& filename, ESfMData partFlag, bool incompleteViews)//,
//              EViewIdMethod viewIdMethod, const std::string& viewIdRegex)
{
//  Version version;

  // load flags
  const bool loadViews = (partFlag & VIEWS) == VIEWS;
  const bool loadIntrinsics = (partFlag & INTRINSICS) == INTRINSICS;
//  const bool loadExtrinsics = (partFlag & EXTRINSICS) == EXTRINSICS;
//  const bool loadStructure = (partFlag & STRUCTURE) == STRUCTURE;
//  const bool loadControlPoints = (partFlag & CONTROL_POINTS) == CONTROL_POINTS;
//  const bool loadFeatures = (partFlag & OBSERVATIONS_WITH_FEATURES) == OBSERVATIONS_WITH_FEATURES;
//  const bool loadObservations = loadFeatures || ((partFlag & OBSERVATIONS) == OBSERVATIONS);

    FILE * fp;
    
    fp = fopen ((foldername + filename).c_str(), "r");
    if(!fp) {
        LOG_ERROR("failed to open sfm data file %s", filename.c_str());
    }
    
    std::string jsonStr;
    fseek(fp , 0, SEEK_END);
    size_t size = ftell(fp);
    jsonStr.resize(size);
    rewind(fp);
    fread(&jsonStr[0], 1, size, fp);
    fclose(fp);
    
    cJSON *monitor_json = cJSON_Parse(jsonStr.c_str());
    if (!monitor_json)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr)
        {
            LOG_ERROR("Error before: %s\n", error_ptr);
        }
        
        return false;
    }
    
    IndexT viewId;
    const cJSON* viewsObj = cJSON_GetObjectItemCaseSensitive(monitor_json, "views");
    if(cJSON_IsArray(viewsObj)) {
        const cJSON* viewObj = nullptr;
        cJSON_ArrayForEach(viewObj, viewsObj)
        {
            cJSON* jviewId = cJSON_GetObjectItemCaseSensitive(viewObj, "viewId");
            if(!cJSON_IsString(jviewId)) {
                LOG_ERROR("Invalid viewId!");
                return false;
            }
            
            cJSON* jposeId = cJSON_GetObjectItemCaseSensitive(viewObj, "poseId");
            if(!cJSON_IsString(jposeId)) {
                LOG_ERROR("Invalid poseId!");
                return false;
            }
            
            cJSON* jintrinsicId = cJSON_GetObjectItemCaseSensitive(viewObj, "intrinsicId");
            if(!cJSON_IsString(jintrinsicId)) {
                LOG_ERROR("Invalid intrinsicId!");
                return false;
            }
            
            cJSON* jwidth = cJSON_GetObjectItemCaseSensitive(viewObj, "width");
            if(!cJSON_IsString(jwidth)) {
                LOG_ERROR("Invalid width!");
                return false;
            }
            
            cJSON* jheight = cJSON_GetObjectItemCaseSensitive(viewObj, "height");
            if(!cJSON_IsString(jheight)) {
                LOG_ERROR("Invalid height!");
                return false;
            }
            
            cJSON* jpath = cJSON_GetObjectItemCaseSensitive(viewObj, "path");
            if(!cJSON_IsString(jpath)) {
                LOG_ERROR("Invalid image path!");
                return false;
            }
         
            using namespace sfmData;
            viewId = (IndexT)std::stoul(cJSON_GetStringValue(jviewId));
            
            std::vector<uint8_t> image_buffer;
            int w,h;
            loadpng(image_buffer, cJSON_GetStringValue(jpath), w, h);
            
            assert(w == (IndexT)std::stoul(cJSON_GetStringValue(jwidth)));
            assert(h == (IndexT)std::stoul(cJSON_GetStringValue(jheight)));
            
            auto&& pView = std::make_shared<View>(
                                                  viewId,
                                                  (IndexT)std::stoul(cJSON_GetStringValue(jintrinsicId)),
                                                  (IndexT)std::stoul(cJSON_GetStringValue(jposeId)),
                                                  (IndexT)std::stoul(cJSON_GetStringValue(jwidth)),
                                                  (IndexT)std::stoul(cJSON_GetStringValue(jheight)),
                                                  image_buffer
                                                  );
            
            
            sfmData.views.insert(std::make_pair(viewId, pView));
        }
    }
    
    const cJSON* intrinsicsObj = cJSON_GetObjectItemCaseSensitive(monitor_json, "intrinsics");
    if(cJSON_IsArray(intrinsicsObj)) {
        const cJSON* intrinsicObj = nullptr;
        cJSON_ArrayForEach(intrinsicObj, intrinsicsObj)
        {
            cJSON* jintrinsicId = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "intrinsicId");
            if(!cJSON_IsString(jintrinsicId)) {
                LOG_ERROR("Invalid intrinsicId!");
                return false;
            }
            
            cJSON* jwidth = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "width");
            if(!cJSON_IsString(jwidth)) {
                LOG_ERROR("Invalid width!");
                return false;
            }
            
            cJSON* jheight = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "height");
            if(!cJSON_IsString(jheight)) {
                LOG_ERROR("Invalid height!");
                return false;
            }
            
            cJSON* jsensorWidth = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "sensorWidth");
            if(!cJSON_IsString(jsensorWidth)) {
                LOG_ERROR("Invalid sensorWidth!");
                return false;
            }
            
//            cJSON* jsensorHeight = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "sensorHeight");
//            if(!cJSON_IsString(jsensorHeight)) {
//                LOG_ERROR("Invalid sensorHeight!");
//                return false;
//            }
            
            cJSON* jfocalLength = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "focalLength");
            if(!cJSON_IsString(jfocalLength)) {
                LOG_ERROR("Invalid focalLength!");
            }
            
            cJSON* jpixelRatio = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "pixelRatio");
            if(!cJSON_IsString(jpixelRatio)) {
                LOG_ERROR("Invalid pixelRatio!");
            }
            
            cJSON* jtype = cJSON_GetObjectItemCaseSensitive(intrinsicObj, "type");
            if(!cJSON_IsString(jtype)) {
                LOG_ERROR("Invalid type!");
            }

            //TODO: check
            if(std::string(cJSON_GetStringValue(jtype)) == "radial3") {
                const double fmm = std::stod(cJSON_GetStringValue(jfocalLength));
                // pixelRatio field was actually storing the focalRatio before version 1.2.5
                const double focalRatio = std::stod(cJSON_GetStringValue(jpixelRatio));
                const double sensorWidth = std::stod(cJSON_GetStringValue(jsensorWidth));
                const double width = std::stod(cJSON_GetStringValue(jwidth));
                const double height = std::stod(cJSON_GetStringValue(jheight));
                const double fx = (fmm / sensorWidth) * width;
                const double fy = fx / focalRatio;
                auto pIntrinsic = std::make_shared<camera::PinholeRadialK3>(width, height, fx, fy);
                
                sfmData.intrinsics.insert(std::make_pair(viewId, pIntrinsic));
            }
        }
    }
    
    cJSON_Delete(monitor_json);

    
    
//    char* sjson = cJSON_Print(monitor);
//    if (sjson == NULL)
//    {
//      LOG_ERROR("Failed to print monitor.\n");
//    }
//    fprintf(fp, "%s", sjson);
    
    fclose(fp);
    
//  // main tree
//  bpt::ptree fileTree;
//
//  // read the json file and initialize the tree
//  bpt::read_json(filename, fileTree);
//
//  // version
//  {
//    Vec3i v;
//    loadMatrix("version", v, fileTree);
//    version = v;
//  }
//
//  // folders
//  if(fileTree.count("featuresFolders"))
//    for(bpt::ptree::value_type& featureFolderNode : fileTree.get_child("featuresFolders"))
//      sfmData.addFeaturesFolder(featureFolderNode.second.get_value<std::string>());
//
//  if(fileTree.count("matchesFolders"))
//    for(bpt::ptree::value_type& matchingFolderNode : fileTree.get_child("matchesFolders"))
//      sfmData.addMatchesFolder(matchingFolderNode.second.get_value<std::string>());
//
//  // intrinsics
//  if(loadIntrinsics && fileTree.count("intrinsics"))
//  {
//    sfmData::Intrinsics& intrinsics = sfmData.getIntrinsics();
//
//    for(bpt::ptree::value_type &intrinsicNode : fileTree.get_child("intrinsics"))
//    {
//      IndexT intrinsicId;
//      std::shared_ptr<camera::IntrinsicBase> intrinsic;
//
//      loadIntrinsic(version, intrinsicId, intrinsic, intrinsicNode.second);
//
//      intrinsics.emplace(intrinsicId, intrinsic);
//    }
//  }
//
//  // views
//  if(loadViews && fileTree.count("views"))
//  {
//    sfmData::Views& views = sfmData.getViews();
//
//    if(incompleteViews)
//    {
//      // store incomplete views in a vector
//      std::vector<sfmData::View> incompleteViews(fileTree.get_child("views").size());
//
//      int viewIndex = 0;
//      for(bpt::ptree::value_type &viewNode : fileTree.get_child("views"))
//      {
//        loadView(incompleteViews.at(viewIndex), viewNode.second);
//        ++viewIndex;
//      }
//
//      // update incomplete views
//      #pragma omp parallel for
//      for(int i = 0; i < incompleteViews.size(); ++i)
//      {
//        sfmData::View& v = incompleteViews.at(i);
//
//        // if we have the intrinsics and the view has an valid associated intrinsics
//        // update the width and height field of View (they are mirrored)
//        if (loadIntrinsics && v.getIntrinsicId() != UndefinedIndexT)
//        {
//          const auto intrinsics = sfmData.getIntrinsicPtr(v.getIntrinsicId());
//
//          if(intrinsics == nullptr)
//          {
//            throw std::logic_error("View " + std::to_string(v.getViewId())
//                                   + " has a intrinsics id " +std::to_string(v.getIntrinsicId())
//                                   + " that cannot be found or the intrinsics are not correctly "
//                                     "loaded from the json file.");
//          }
//
//          v.setWidth(intrinsics->w());
//          v.setHeight(intrinsics->h());
//        }
//        updateIncompleteView(incompleteViews.at(i), viewIdMethod, viewIdRegex);
//      }
//
//      // copy complete views in the SfMData views map
//      for(const sfmData::View& view : incompleteViews)
//        views.emplace(view.getViewId(), std::make_shared<sfmData::View>(view));
//    }
//    else
//    {
//      // store directly in the SfMData views map
//      for(bpt::ptree::value_type &viewNode : fileTree.get_child("views"))
//      {
//        sfmData::View view;
//        loadView(view, viewNode.second);
//        views.emplace(view.getViewId(), std::make_shared<sfmData::View>(view));
//      }
//    }
//  }
//
//  // extrinsics
//  if(loadExtrinsics)
//  {
//    // poses
//    if(fileTree.count("poses"))
//    {
//      sfmData::Poses& poses = sfmData.getPoses();
//
//      for(bpt::ptree::value_type &poseNode : fileTree.get_child("poses"))
//      {
//        bpt::ptree& poseTree = poseNode.second;
//        sfmData::CameraPose pose;
//
//        loadCameraPose("pose", pose, poseTree);
//
//        poses.emplace(poseTree.get<IndexT>("poseId"), pose);
//      }
//    }
//
//    // rigs
//    if(fileTree.count("rigs"))
//    {
//      sfmData::Rigs& rigs = sfmData.getRigs();
//
//      for(bpt::ptree::value_type &rigNode : fileTree.get_child("rigs"))
//      {
//        IndexT rigId;
//        sfmData::Rig rig;
//
//        loadRig(rigId, rig, rigNode.second);
//
//        rigs.emplace(rigId, rig);
//      }
//    }
//  }
//
//  // structure
//  if(loadStructure && fileTree.count("structure"))
//  {
//    sfmData::Landmarks& structure = sfmData.getLandmarks();
//
//    for(bpt::ptree::value_type &landmarkNode : fileTree.get_child("structure"))
//    {
//      IndexT landmarkId;
//      sfmData::Landmark landmark;
//
//      loadLandmark(landmarkId, landmark, landmarkNode.second, loadObservations, loadFeatures);
//
//      structure.emplace(landmarkId, landmark);
//    }
//  }
//
//  // control points
//  if(loadControlPoints && fileTree.count("controlPoints"))
//  {
//    sfmData::Landmarks& controlPoints = sfmData.getControlPoints();
//
//    for(bpt::ptree::value_type &landmarkNode : fileTree.get_child("controlPoints"))
//    {
//      IndexT landmarkId;
//      sfmData::Landmark landmark;
//
//      loadLandmark(landmarkId, landmark, landmarkNode.second);
//
//      controlPoints.emplace(landmarkId, landmark);
//    }
//  }

  return true;
}

} // namespace sfmDataIO

