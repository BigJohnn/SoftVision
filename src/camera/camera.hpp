// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <camera/cameraCommon.hpp>
#include <camera/IntrinsicBase.hpp>
#include <camera/Pinhole.hpp>
#include <camera/PinholeRadial.hpp>
//#include <camera/Pinhole3DE.hpp>
//#include <camera/PinholeBrown.hpp>
//#include <camera/PinholeFisheye.hpp>
//#include <camera/PinholeFisheye1.hpp>
#include <camera/Equidistant.hpp>
//#include <camera/EquidistantRadial.hpp>
//#include <camera/cameraUndistortImage.hpp>

namespace camera {

inline std::shared_ptr<IntrinsicBase> createIntrinsic(EINTRINSIC intrinsicType,
    unsigned int w = 0, unsigned int h = 0,
    double focalLengthPixX = 0.0, double focalLengthPixY = 0.0,
    double offsetX = 0.0, double offsetY = 0.0)
{
    switch (intrinsicType)
    {
    case EINTRINSIC::PINHOLE_CAMERA:
        return std::make_shared<Pinhole>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
    case EINTRINSIC::PINHOLE_CAMERA_RADIAL1:
        return std::make_shared<PinholeRadialK1>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
    case EINTRINSIC::PINHOLE_CAMERA_RADIAL3:
        return std::make_shared<PinholeRadialK3>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_3DERADIAL4:
//        return std::make_shared<Pinhole3DERadial4>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_BROWN:
//        return std::make_shared<PinholeBrownT2>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_FISHEYE:
//        return std::make_shared<PinholeFisheye>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_FISHEYE1:
//        return std::make_shared<PinholeFisheye1>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_3DEANAMORPHIC4:
//        return std::make_shared<Pinhole3DEAnamorphic4>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
//    case EINTRINSIC::PINHOLE_CAMERA_3DECLASSICLD:
//        return std::make_shared<Pinhole3DEClassicLD>(w, h, focalLengthPixX, focalLengthPixY, offsetX, offsetY);
    case EINTRINSIC::EQUIDISTANT_CAMERA:
        return std::make_shared<EquiDistant>(w, h, focalLengthPixX, offsetX, offsetY);
//    case EINTRINSIC::EQUIDISTANT_CAMERA_RADIAL3:
//        return std::make_shared<EquiDistantRadialK3>(w, h, focalLengthPixX, offsetX, offsetY);
    case EINTRINSIC::UNKNOWN:
    case EINTRINSIC::VALID_PINHOLE:
    case EINTRINSIC::VALID_EQUIDISTANT:
    case EINTRINSIC::VALID_CAMERA_MODEL:
        break;
    }
    throw std::out_of_range("Unrecognized Intrinsic Enum");
}

} // namespace camera

