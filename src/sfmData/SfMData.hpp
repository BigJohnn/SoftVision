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
#include <CameraPose.hpp>
#include <sfmData/Landmark.hpp>

namespace camera {
class IntrinsicBase;
}

namespace sfmData {

class View;
class CameraPose;

/// Define a collection of View
using Views = std::map<IndexT, std::shared_ptr<View> >;

/// Define a collection of Pose (indexed by view.getPoseId())
using Poses = HashMap<IndexT, CameraPose>;

/// Define a collection of IntrinsicParameter (indexed by view.getIntrinsicId())
using Intrinsics = std::map<IndexT, std::shared_ptr<camera::IntrinsicBase> >;

/// Define a collection of landmarks are indexed by their TrackId
using Landmarks = HashMap<IndexT, Landmark>;

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
//    /// 2D Constraints
//    Constraints2D constraints2d;
//    /// Rotation priors
//    RotationPriors rotationpriors;
    
    
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
     * @brief Get poses
     * @return poses
    */
    const Poses& getPoses() const {return _poses;}
    Poses& getPoses() {return _poses;}
//    ~SfMData();
    
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
    
    
private:
    /// Considered poses (indexed by view.getPoseId())
    Poses _poses;
};
}
#endif /* SfMData_hpp */
