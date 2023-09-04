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

}
