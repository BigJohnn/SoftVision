// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <mvsData/StaticVector.hpp>
#include <mvsUtils/MultiViewParams.hpp>
#include <mesh/Mesh.hpp>



class Point3d;

namespace mesh {


void filterLargeEdgeTriangles(Mesh* me, float avelthr);

void meshPostProcessing(Mesh*& inout_mesh, StaticVector<StaticVector<int>>& inout_ptsCams, mvsUtils::MultiViewParams& mp,
                      const std::string& debugFolderName,
                      StaticVector<Point3d>* hexahsToExcludeFromResultingMesh, Point3d* hexah);

} // namespace mesh

