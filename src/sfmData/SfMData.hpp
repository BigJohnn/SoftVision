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

namespace camera {
class IntrinsicBase;
}

namespace sfmData {

class View;
class CameraPose;

/// Define a collection of View
using Views = std::map<IndexT, std::shared_ptr<View> >;

/// Define a collection of Pose (indexed by view.getPoseId())
//using Poses = HashMap<IndexT, CameraPose>;

/// Define a collection of IntrinsicParameter (indexed by view.getIntrinsicId())
using Intrinsics = std::map<IndexT, std::shared_ptr<camera::IntrinsicBase> >;

//class IntrinsicBase;
class SfMData{
public:
    /// Considered views
    Views views;
    /// Considered camera intrinsics (indexed by view.getIntrinsicId())
    Intrinsics intrinsics;
    
    SfMData() = default;
    
    /**
         * @brief Get views
         * @return views
         */
        const Views& getViews() const {return views;}
        Views& getViews() {return views;}
    
    /**
         * @brief Get intrinsics
         * @return intrinsics
         */
        const Intrinsics& getIntrinsics() const {return intrinsics;}
        Intrinsics& getIntrinsics() {return intrinsics;}
    
    /**
     * @brief Get poses
     * @return poses
    */
//    const Poses& getPoses() const {return _poses;}
//    Poses& getPoses() {return _poses;}
//    ~SfMData();
    
private:
    /// Considered poses (indexed by view.getPoseId())
//    Poses _poses;
};
}
#endif /* SfMData_hpp */
