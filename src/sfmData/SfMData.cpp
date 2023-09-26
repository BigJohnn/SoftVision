//
//  SfMData.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#include "SfMData.hpp"
#include "View.hpp"
#include "IntrinsicBase.hpp"
#include <SoftVisionLog.h>

namespace sfmData{

std::vector<std::string> SfMData::getFeaturesFolders() const
{
    return _featuresFolders;
}

std::vector<std::string> SfMData::getMatchesFolders() const
{
    return _matchesFolders;
}

std::vector<std::string> SfMData::getRelativeFeaturesFolders() const
{
    LOG_DEBUG("TODO: check this");
    return _featuresFolders;
}

std::vector<std::string> SfMData::getRelativeMatchesFolders() const
{
    LOG_DEBUG("TODO: check this");
    return _matchesFolders;
}



void SfMData::addFeaturesFolders(const std::vector<std::string>& folders)
{
    _featuresFolders.insert(_featuresFolders.begin(), folders.begin(), folders.end());
}

void SfMData::addMatchesFolders(const std::vector<std::string>& folders)
{
    _matchesFolders.insert(_matchesFolders.begin(), folders.begin(), folders.end());
}

void SfMData::setAbsolutePath(const std::string& path)
{
    // Change internal absolute path
    _absolutePath = path;
}

std::set<IndexT> SfMData::getValidViews() const
{
    std::set<IndexT> valid_idx;
    for (Views::const_iterator it = views.begin(); it != views.end(); ++it)
    {
        const View * v = it->second.get();
        if (isPoseAndIntrinsicDefined(v))
        {
            valid_idx.insert(v->getViewId());
        }
    }
    return valid_idx;
}

bool SfMData::isPoseAndIntrinsicDefined(const View* view) const
{
    if (view == nullptr)
        return false;
    return (
        view->getIntrinsicId() != UndefinedIndexT &&
        view->getPoseId() != UndefinedIndexT &&
//        (!view->isPartOfRig() || view->isPoseIndependant() || getRigSubPose(*view).status != ERigSubPoseStatus::UNINITIALIZED) &&
        intrinsics.find(view->getIntrinsicId()) != intrinsics.end() &&
        _poses.find(view->getPoseId()) != _poses.end()
    );
}

std::set<IndexT> SfMData::getReconstructedIntrinsics() const
{
    std::set<IndexT> valid_idx;
    for (Views::const_iterator it = views.begin(); it != views.end(); ++it)
    {
        const View * v = it->second.get();
        if (isPoseAndIntrinsicDefined(v))
        {
            valid_idx.insert(v->getIntrinsicId());
        }
    }
    return valid_idx;
}

void SfMData::setPose(const View& view, const CameraPose& absolutePose)
{
    // const bool knownPose = existsPose(view);
    CameraPose& viewPose = _poses[view.getPoseId()];

    // Pose dedicated for this view (independant from rig, even if it is potentially part of a rig)
    if (view.isPoseIndependant())
    {
        viewPose = absolutePose;
        return;
    }

    // Initialized rig
//    if (view.getRigId() != UndefinedIndexT)
//    {
//        const Rig& rig = _rigs.at(view.getRigId());
//        RigSubPose& subPose = getRigSubPose(view);
//
//        viewPose.setTransform(subPose.pose.inverse() * absolutePose.getTransform());
//
//        if (absolutePose.isLocked())
//        {
//            viewPose.lock();
//        }
//
//        return;
//    }

    throw std::runtime_error("SfMData::setPose: dependant view pose not part of an initialized rig.");
}

}
