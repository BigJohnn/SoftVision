//
//  SfMData.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef SfMData_hpp
#define SfMData_hpp

#include <map>
#include <memory>

#include <common/types.h>
#include <sfmData/CameraPose.hpp>
#include <sfmData/Landmark.hpp>
#include <sfmData/View.hpp>
#include <sfmData/Constraint2D.hpp>
#include <sfmData/RotationPrior.hpp>

namespace camera {
class IntrinsicBase;
}

namespace sfmData {

class CameraPose;

/// Define a collection of View
using Views = std::map<IndexT, std::shared_ptr<View> >;

/// Define a collection of Pose (indexed by view.getPoseId())
using Poses = HashMap<IndexT, CameraPose>;

/// Define a collection of IntrinsicParameter (indexed by view.getIntrinsicId())
using Intrinsics = std::map<IndexT, std::shared_ptr<camera::IntrinsicBase> >;

/// Define a collection of landmarks are indexed by their TrackId
using Landmarks = HashMap<IndexT, Landmark>;

///Define a collection of constraints
using Constraints2D = std::vector<Constraint2D>;

///Define a collection of rotation priors
using RotationPriors = std::vector<RotationPrior>;

//class IntrinsicBase;
class SfMData{
public:
    /// Considered views
    Views views;
    /// Considered camera intrinsics (indexed by view.getIntrinsicId())
    Intrinsics intrinsics;
    /// Structure (3D points with their 2D observations)
    Landmarks structure;
/// Controls points (stored as Landmarks (id_feat has no meaning here))
    Landmarks control_points;
//    /// Uncertainty per pose
//    PosesUncertainty _posesUncertainty;
//    /// Uncertainty per landmark
//    LandmarksUncertainty _landmarksUncertainty;
    /// 2D Constraints
    Constraints2D constraints2d;
    /// Rotation priors
    RotationPriors rotationpriors;
    
    
    SfMData() = default;
    
    /**
     * @brief Get views
     * @return views
     */
    const Views& getViews() const {return views;}
    Views& getViews() {return views;}
    
    /**
     * @brief Gives the view of the input view id.
     * @param[in] viewId The given view id
     * @return the corresponding view reference
     */
    View& getView(IndexT viewId)
    {
        return *(views.at(viewId));
    }

    /**
     * @brief Gives the view of the input view id.
     * @param[in] viewId The given view id
     * @return the corresponding view reference
     */
    const View& getView(IndexT viewId) const
    {
        return *(views.at(viewId));
    }
    
    /**
     * @brief Get intrinsics
     * @return intrinsics
     */
    const Intrinsics& getIntrinsics() const {return intrinsics;}
    Intrinsics& getIntrinsics() {return intrinsics;}
    
    /**
     * @brief List the view indexes that have valid camera intrinsic and pose.
     * @return view indexes list
     */
    std::set<IndexT> getValidViews() const;
    
    /**
     * @brief List the intrinsic indexes that have valid camera intrinsic and pose.
     * @return intrinsic indexes list
     */
    std::set<IndexT> getReconstructedIntrinsics() const;
    
    /**
     * @brief Get poses
     * @return poses
    */
    const Poses& getPoses() const {return _poses;}
    Poses& getPoses() {return _poses;}
//    ~SfMData();
    
    /**
     * @brief Set the given pose for the given view
     * if the view is part of a rig, this method update rig pose/sub-pose
     * @param[in] view The given view
     * @param[in] pose The given pose
     */
    void setPose(const View& view, const CameraPose& pose);
    
    /**
     * @brief Check if the given view have defined intrinsic and pose
     * @param[in] view The given view
     * @return true if intrinsic and pose defined
     */
    bool isPoseAndIntrinsicDefined(const View* view) const;
    
    /**
     * @brief Check if the given view have defined intrinsic and pose
     * @param[in] viewID The given viewID
     * @return true if intrinsic and pose defined
     */
    bool isPoseAndIntrinsicDefined(IndexT viewId) const
    {
        return isPoseAndIntrinsicDefined(views.at(viewId).get());
    }
    
    /**
     * @brief Get landmarks
     * @return landmarks
     */
    const Landmarks& getLandmarks() const {return structure;}
    Landmarks& getLandmarks() {return structure;}
    
    /**
     * @brief Get Constraints2D
     * @return Constraints2D
     */
    const Constraints2D& getConstraints2D() const {return constraints2d;}
    Constraints2D& getConstraints2D() {return constraints2d;}
    
    /**
     * @brief Get RotationPriors
     * @return RotationPriors
     */
    const RotationPriors& getRotationPriors() const {return rotationpriors;}
    RotationPriors& getRotationPriors() {return rotationpriors;}

    /**
     * @brief Get control points
     * @return control points
     */
    const Landmarks& getControlPoints() const {return control_points;}
    Landmarks& getControlPoints() {return control_points;}
    
    /**
     * @brief Gives the pose of the input view. If this view is part of a rig, it returns rigPose + rigSubPose.
     * @param[in] view The given view
     *
     * @warning: This function returns a CameraPose (a temporary object and not a reference),
     *           because in the RIG context, this pose is the composition of the rig pose and the sub-pose.
     */
    CameraPose getPose(const View& view) const
    {
        // check the view has valid pose / rig etc
        if (view.isPoseIndependant())
        {
            return _poses.at(view.getPoseId());
        }

//        // get the pose of the rig
//        CameraPose pose = getRigPose(view);
//
//        // multiply rig pose by camera subpose
//        pose.setTransform(getRigSubPose(view).pose * pose.getTransform());

//        return pose;
    }
    
    /**
     * @brief  Gives the pose with the given pose id.
     * @param[in] poseId The given pose id
     */
    const CameraPose& getAbsolutePose(IndexT poseId) const
    {
        return _poses.at(poseId);
    }
    
    /**
     * @brief Erase yhe pose for the given poseId
     * @param[in] poseId The given poseId
     * @param[in] noThrow If false, throw exception if no pose found
     */
    void erasePose(IndexT poseId, bool noThrow = false)
    {
        auto it =_poses.find(poseId);
        if (it != _poses.end())
            _poses.erase(it);
        else if (!noThrow)
            throw std::out_of_range(std::string("Can't erase unfind pose ") + std::to_string(poseId));
    }
    
    std::set<feature::EImageDescriberType> getLandmarkDescTypes() const
    {
        std::set<feature::EImageDescriberType> output;
        for (auto s : getLandmarks())
        {
            output.insert(s.second.descType);
        }
        return output;
    }
    
    std::map<feature::EImageDescriberType, int> getLandmarkDescTypesUsages() const
    {
        std::map<feature::EImageDescriberType, int> output;
        for (auto s : getLandmarks())
        {
            if (output.find(s.second.descType) == output.end())
            {
                output[s.second.descType] = 1;
            }
            else
            {
                ++output[s.second.descType];
            }
        }
        return output;
    }

    /**
     * @brief Return a pointer to an intrinsic if available or nullptr otherwise.
     * @param[in] intrinsicId
     */
    const camera::IntrinsicBase* getIntrinsicPtr(IndexT intrinsicId) const
    {
        if (intrinsics.count(intrinsicId))
            return intrinsics.at(intrinsicId).get();
        return nullptr;
    }
    
    /**
     * @brief Return a shared pointer to an intrinsic if available or nullptr otherwise.
     * @param[in] intrinsicId
     */
    std::shared_ptr<camera::IntrinsicBase> getIntrinsicsharedPtr(IndexT intrinsicId)
    {
        if(intrinsics.count(intrinsicId))
            return intrinsics.at(intrinsicId);
        return nullptr;
    }

    /**
     * @brief Return a shared pointer to an intrinsic if available or nullptr otherwise.
     * @param[in] intrinsicId
     */
    const std::shared_ptr<camera::IntrinsicBase> getIntrinsicsharedPtr(IndexT intrinsicId) const
    {
        if(intrinsics.count(intrinsicId))
            return intrinsics.at(intrinsicId);
        return nullptr;
    }
    
    /**
     * @brief Get absolute features folder paths
     * @return features folders paths
     */
    std::vector<std::string> getFeaturesFolders() const;

    /**
     * @brief Get absolute matches folder paths
     * @return matches folder paths
     */
    std::vector<std::string> getMatchesFolders() const;
    
    /**
     * @brief Add the given \p folder to features folders.
     * @note If SfmData's absolutePath has been set,
     *       an absolute path will be converted to a relative one.
     * @param[in] folder path to a folder containing features
     */
    inline void addFeaturesFolder(const std::string& folder)
    {
        addFeaturesFolders({folder});
    }

    /**
     * @brief Add the given \p folders to features folders.
     * @note If SfmData's absolutePath has been set,
     *       absolute paths will be converted to relative ones.
     * @param[in] folders paths to folders containing features
     */
    void addFeaturesFolders(const std::vector<std::string>& folders);

    /**
     * @brief Add the given \p folder to matches folders.
     * @note If SfmData's absolutePath has been set,
     *       an absolute path will be converted to a relative one.
     * @param[in] folder path to a folder containing matches
     */
    inline void addMatchesFolder(const std::string& folder)
    {
        addMatchesFolders({folder});
    }

    /**
     * @brief Add the given \p folders to matches folders.
     * @note If SfmData's absolutePath has been set,
     *       absolute paths will be converted to relative ones.
     * @param[in] folders paths to folders containing matches
     */
    void addMatchesFolders(const std::vector<std::string>& folders);

    /**
     * @brief Replace the current features folders by the given ones.
     * @note If SfmData's absolutePath has been set,
     *       absolute paths will be converted to relative ones.
     * @param[in] folders paths to folders containing features
     */
    inline void setFeaturesFolders(const std::vector<std::string>& folders)
    {
        _featuresFolders.clear();
        addFeaturesFolders(folders);
    }

    /**
     * @brief Replace the current matches folders by the given ones.
     * @note If SfmData's absolutePath has been set,
     *       absolute paths will be converted to relative ones.
     * @param[in] folders paths to folders containing matches
     */
    inline void setMatchesFolders(const std::vector<std::string>& folders)
    {
        _matchesFolders.clear();
        addMatchesFolders(folders);
    }

    /**
     * @brief Set the SfMData file absolute path.
     * @note Internal relative features/matches folders will be remapped
     *       to be relative to the new absolute \p path.
     * @param[in] path The absolute path to the SfMData file folder
     */
    void setAbsolutePath(const std::string& path);

    inline const std::string getRootPath() const
    {
        return _rootPath;
    }
    
    /**
     * @brief Get a set of views keys
     * @return set of views keys
     */
    std::set<IndexT> getViewsKeys() const
    {
        std::set<IndexT> viewKeys;
        for (auto v: views)
            viewKeys.insert(v.first);
        return viewKeys;
    }
    
private:
    /// Absolute path to the SfMData file (should not be saved)
    std::string _absolutePath;
    /// root path, **/tmp/
    std::string _rootPath;
    /// Features folders path
    std::vector<std::string> _featuresFolders;
    /// Matches folders path
    std::vector<std::string> _matchesFolders;
    
    /// Considered poses (indexed by view.getPoseId())
    Poses _poses;
};
}
#endif /* SfMData_hpp */
