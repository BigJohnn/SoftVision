//
//  SfMData.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#include "SfMData.hpp"
#include "View.hpp"
#include "IntrinsicBase.hpp"

namespace sfmData{

/**
 * @brief Convert paths in \p folders to absolute paths using \p absolutePath parent folder as base.
 * @param[in] folders list of paths to convert
 * @param[in] absolutePath filepath which parent folder should be used as base for absolute path conversion
 * @return the list of converted absolute paths or input folder if absolutePath is empty
 */
std::vector<std::string> toAbsoluteFolders(const std::vector<std::string>& folders, const std::string& absolutePath)
{
    // If absolute path is not set, return input folders
    if (absolutePath.empty())
        return folders;
    // Else, convert relative paths to absolute paths
    std::vector<std::string> absolutePaths;
    absolutePaths.reserve(folders.size());
    for (const auto& folder: folders)
    {
        const fs::path f = fs::absolute(folder, fs::path(absolutePath).parent_path());
        if (fs::exists(f))
        {
            // fs::canonical can only be used if the path exists
            absolutePaths.push_back(fs::canonical(f).string());
        }
        else
        {
            absolutePaths.push_back(f.string());
        }
    }
    return absolutePaths;
}

/**
 * @brief Add paths contained in \p folders to \p dst as relative paths to \p absolutePath.
 *        Paths already present in \p dst are omitted.
 * @param[in] dst list in which paths should be added
 * @param[in] folders paths to add to \p dst as relative folders
 * @param[in] absolutePath filepath which parent folder should be used as base for relative path conversions
 */
void addAsRelativeFolders(std::vector<std::string>& dst, const std::vector<std::string>& folders, const std::string& absolutePath)
{
    for (auto folderPath: folders)
    {
        // If absolutePath is set, convert to relative path
        if (!absolutePath.empty() && fs::path(folderPath).is_absolute())
        {
            folderPath = fs::relative(folderPath, fs::path(absolutePath).parent_path()).string();
        }
        // Add path only if not already in dst
        if (std::find(dst.begin(), dst.end(), folderPath) == dst.end())
        {
            dst.emplace_back(folderPath);
        }
    }
}

std::vector<std::string> SfMData::getFeaturesFolders() const
{
    return toAbsoluteFolders(_featuresFolders, _absolutePath);
}

std::vector<std::string> SfMData::getMatchesFolders() const
{
    return toAbsoluteFolders(_matchesFolders, _absolutePath);
}

void SfMData::addFeaturesFolders(const std::vector<std::string>& folders)
{
    addAsRelativeFolders(_featuresFolders, folders, _absolutePath);
}

void SfMData::addMatchesFolders(const std::vector<std::string>& folders)
{
    addAsRelativeFolders(_matchesFolders, folders, _absolutePath);
}

void SfMData::setAbsolutePath(const std::string& path)
{
    // Get absolute path to features/matches folders
    const std::vector<std::string> featuresFolders = getFeaturesFolders();
    const std::vector<std::string> matchesFolders = getMatchesFolders();
    // Change internal absolute path
    _absolutePath = path;
    // Re-set features/matches folders
    // They will be converted back to relative paths based on updated _absolutePath
    setFeaturesFolders(featuresFolders);
    setMatchesFolders(matchesFolders);
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

std::vector<std::string> SfMData::getFeaturesFolders() const
{
    return std::vector<std::string>{};
}

std::vector<std::string> SfMData::getMatchesFolders() const
{
    return std::vector<std::string>{};
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
