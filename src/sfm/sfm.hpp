// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// SfM

#include <sfm/filters.hpp>
#include <sfm/FrustumFilter.hpp>
#include <sfm/BundleAdjustment.hpp>
#include <sfm/BundleAdjustmentCeres.hpp>
#include <sfm/LocalBundleAdjustmentGraph.hpp>
#include <sfm/generateReport.hpp>
#include <sfm/sfmFilters.hpp>
#include <sfm/sfmTriangulation.hpp>

// SfM pipeline

#include <sfm/pipeline/ReconstructionEngine.hpp>
#include <sfm/pipeline/pairwiseMatchesIO.hpp>
#include <sfm/pipeline/RelativePoseInfo.hpp>
//#include <sfm/pipeline/global/reindexGlobalSfM.hpp>
//#include <sfm/pipeline/global/ReconstructionEngine_globalSfM.hpp>
//#include <sfm/pipeline/panorama/ReconstructionEngine_panorama.hpp>
#include <sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.hpp>
#include <sfm/pipeline/structureFromKnownPoses/StructureEstimationFromKnownPoses.hpp>
#include <sfm/pipeline/localization/SfMLocalizer.hpp>
#include <sfm/pipeline/localization/SfMLocalizationSingle3DTrackObservationDatabase.hpp>

