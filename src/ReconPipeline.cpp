//
//  camerainit.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/13.
//

#include <ReconPipeline.h>
//#include <featureEngine/FeatureExtractor.hpp>

//#include <feature/feature.hpp>
#include <utils/YuvImageProcessor.h>
#include "PngUtils.h"

#include <sfmData/SfMData.hpp>
#include <sfmData/View.hpp>
#include <camera/PinholeRadial.hpp>
#include <featureEngine/FeatureExtractor.hpp>
#include <system/Timer.hpp>
#include <SoftVisionLog.h>

#include <imageMatching/ImageMatching.hpp>
#include "voctree/descriptorLoader.hpp"

#include <matching/IndMatch.hpp>
#include <matching/matcherType.hpp>
#include <matching/matchesFiltering.hpp>

#include <matchingImageCollection/IImageCollectionMatcher.hpp>
#include <matchingImageCollection/matchingCommon.hpp>
#include <matchingImageCollection/GeometricFilterType.hpp>
#include <matchingImageCollection/GeometricFilter.hpp>
#include <matchingImageCollection/GeometricFilterMatrix_F_AC.hpp>

#include <sfm/pipeline/structureFromKnownPoses/StructureEstimationFromKnownPoses.hpp>

#include <robustEstimation/estimators.hpp>


std::vector<std::vector<uint8_t>> ReconPipeline::m_cachedBuffers;

#ifdef SOFTVISION_DEBUG
#include <thread>
#endif

void getStatsMap(const matching::PairwiseMatches& map)
{
#ifdef ALICEVISION_DEBUG_MATCHING
  std::map<int,int> stats;
  for(const auto& imgMatches: map)
  {
    for(const auto& featMatchesPerDesc: imgMatches.second)
    {
      for(const matching::IndMatch& featMatches: featMatchesPerDesc.second)
      {
        int d = std::floor(featMatches._distance / 1000.0);
        if( stats.find(d) != stats.end() )
          stats[d] += 1;
        else
          stats[d] = 1;
      }
    }
  }
  for(const auto& stat: stats)
  {
    LOG_DEBUG("%s\t%s", std::to_string(stat.first).c_str(), std::to_string(stat.second).c_str());
  }
#endif
}


ReconPipeline ReconPipeline::GetInstance()
{
    static ReconPipeline pipeline;
    return pipeline;
}

ReconPipeline::~ReconPipeline()
{
    delete m_sfmData;
    LOG_DEBUG("ReconPipeline Destruction");
    
    ///TODO: check
    m_cachedBuffers.clear();
    
    delete m_extractor;
}

void ReconPipeline::CameraInit(void) {
    printf("%p Do camera init ...\n", this);
    
    float fov =  45.0f; //degree
    float sensorWidthmm = 36.0f;
    //Construct camera intrinsic ...
    printf("fov == %.2f, sensorWidthmm == %.2f ...\n",fov,sensorWidthmm);
    
    m_sfmData = new sfmData::SfMData();
}

void ReconPipeline::AppendSfMData(uint32_t viewId,
                   uint32_t poseId,
                   uint32_t intrinsicId,
                   uint32_t frameId,
                  uint32_t width,
                  uint32_t height,
                   const uint8_t* bufferData)
{
    
    printf("%p Do AppendSfMData ...\n", this);
    
    printf("ReconPipeline viewId, poseId, intrinsicId: %u %u %u\n", viewId, poseId,intrinsicId);
    printf("ReconPipeline width, height, bufferData: %d %d %p\n",width, height,bufferData);
    
    std::cout<<"[tid] AppendSfMData"<<std::this_thread::get_id()<<std::endl;
    
    std::map<std::string, std::string> metadata_; //TODO: fill this
    
    const int buffer_n_bytes = width * height * 4;
    
    std::vector<uint8_t> buf(buffer_n_bytes, 0);
    m_cachedBuffers.push_back(buf); //TODO: check??
    memcpy(buf.data(), bufferData, buffer_n_bytes);
    
    int width_new, height_new;
    {
        auto* buffer = new uint8_t[width * height * 4];
        Convert2Portrait(width, height, buf.data(), width_new, height_new, buffer);
        
        if(m_outputFolder.empty()) {
            LOG_ERROR("OUTPUT DIR NOT SET!!");
        }
        auto&& folder_name = m_outputFolder.substr(7, m_outputFolder.size() - 7);
        std::string testimg_file_name = folder_name + "test.png";
        write2png(testimg_file_name.c_str(), width_new, height_new, buffer);

//        image::Image<image::RGBAColor> imageRGBA_flipY;
//        uint8_t *buffer_flipY  = new uint8_t[view.getWidth() * view.getHeight() * 4];
        FlipY(width_new, height_new, buffer, buf.data());
        delete buffer;
    }
    
    auto pView = std::make_shared<sfmData::View>(viewId,
                                                 intrinsicId,
                                                 poseId,
                                                  width_new,
                                                  height_new,
                                                  buf,
                                                  metadata_);
    
    pView->setFrameId(frameId);
    auto&& views = m_sfmData->views;
//    views[IndexT(views.size() - 1)] = pView;
    views.insert(std::make_pair(IndexT(views.size()), pView));
    
    //TODO:
    auto pIntrinsic = std::make_shared<camera::PinholeRadialK3>(width, height);
    auto&& intrinsics = m_sfmData->intrinsics;
    intrinsics[IndexT(intrinsics.size() - 1)] = pIntrinsic;
    
#ifdef SOFTVISION_DEBUG
    printf("views addr ===%p\n", &views);
    
    for(int i=0;i<views.size();i++)
    {
        if(views[i]->getBuffer()){
            LOG_INFO("buffer[%d]: addr %p, %.*s", i,&views[i], 100,views[i]->getBuffer());
        }
        else {
            LOG_INFO("buffer[%d]: NULL!!",i);
        }
    }
#endif
}

void ReconPipeline::SetOutputDataDir(const char* directory)
{
    printf("%p Do SetOutputDataDir ...\n", this);
    m_outputFolder = directory;
}

bool ReconPipeline::FeatureExtraction()
{
#ifdef SOFTVISION_DEBUG
    std::cout<<"[tid] FeatureExtraction"<<std::this_thread::get_id()<<std::endl;
    
    auto&& views = m_sfmData->views;
    printf("views addr ===%p\n", &views);
    
    for(int i=0;i<views.size();i++)
    {
        if(views[i]->getBuffer()){
            LOG_INFO("buffer[%d]:%.*s", i,100,views[i]->getBuffer());
        }
        else {
            LOG_INFO("buffer[%d]: NULL!!",i);
        }
    }
#endif
    
    printf("%p Do FeatureExtraction ...\n", this);
    
    m_extractor = new featureEngine::FeatureExtractor(*m_sfmData);
//    featureEngine::FeatureExtractor extractor(*m_sfmData);
    auto&& extractor = *m_extractor;
    int rangeStart = 0, rangeSize = m_sfmData->views.size();
    extractor.setRange(rangeStart, rangeSize);
    
    m_describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::DSPSIFT);
    {
        std::vector<feature::EImageDescriberType> imageDescriberTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);

        for(const auto& imageDescriberType: imageDescriberTypes)
        {
            std::shared_ptr<feature::ImageDescriber> imageDescriber = feature::createImageDescriber(imageDescriberType);
            
            feature::ConfigurationPreset featDescConfig; //TODO: set this
            imageDescriber->setConfigurationPreset(featDescConfig);
            extractor.addImageDescriber(imageDescriber);
        }
    }
    
    // set maxThreads
    HardwareContext hwc;
    hwc.setUserCoresLimit(8);
    
    // feature extraction routines
    // for each View of the SfMData container:
    // - if regions file exist continue,
    // - if no file, compute features
    {
        system2::Timer timer;
        extractor.setOutputFolder(m_outputFolder);
        extractor.process(hwc, image::EImageColorSpace::SRGB);

        LOG_INFO("Task done in (s):%s " , std::to_string(timer.elapsed()).c_str());
    }
    
    return true;
}

bool ReconPipeline::FeatureMatching()
{
    LOG_INFO("FeatureMatching\n"
             "- Compute putative local feature matches (descriptors matching)\n"
             "- Compute geometric coherent feature matches (robust model estimation from putative matches)\n");
    
    // user optional parameters
    imageMatching::EImageMatchingMethod method = imageMatching::EImageMatchingMethod::SEQUENTIAL_AND_VOCABULARYTREE;
    /// minimal number of images to use the vocabulary tree
    std::size_t minNbImages = 200;
    /// the file containing the list of features
    std::size_t nbMaxDescriptors = 500;
    /// the number of matches to retrieve for each image in Vocabulary Tree Mode
    std::size_t numImageQuery = 50;
    /// the number of neighbors to retrieve for each image in Sequential Mode
    std::size_t numImageQuerySequential = 50;

    //      "Maximum error (in pixels) allowed for features matching guided by geometric information from known camera poses. "
    //            "If set to 0 it lets the ACRansac select an optimal value.")
    double knownPosesGeometricErrorMax = 4.0;
    
    int rangeSize = m_sfmData->views.size();
    int rangeStart = 0;
    
    PairSet pairs;
    std::set<IndexT> filter;

    // We assume that there is only one pair for (I,J) and (J,I)
    pairs = exhaustivePairs(m_sfmData->getViews(), 0, m_sfmData->getViews().size());
    
    if(pairs.empty())
      {
        LOG_INFO("No image pair to match.");
        // if we only compute a selection of matches, we may have no match.
        return m_sfmData->getViews().size() ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    
    LOG_INFO("Number of pairs: %lu" , pairs.size());
    
    // filter creation
    for(const auto& pair: pairs)
    {
    filter.insert(pair.first);
    filter.insert(pair.second);
    }

    matching::PairwiseMatches mapPutativesMatches;
    
    // allocate the right Matcher according the Matching requested method
//    matching::EMatcherType collectionMatcherType = EMatcherType_stringToEnum(nearestMatchingMethod);
//      std::unique_ptr<IImageCollectionMatcher> imageCollectionMatcher = createImageCollectionMatcher(collectionMatcherType, distRatio, crossMatching);
    std::string nearestMatchingMethod = "ANN_L2";
    // allocate the right Matcher according the Matching requested method
    matching::EMatcherType collectionMatcherType = matching::EMatcherType_stringToEnum(nearestMatchingMethod);
    
    
    using namespace matchingImageCollection;
    //"Distance ratio to discard non meaningful matches."
    float distRatio;
//    "Make sure that the matching process is symmetric (same matches for I->J than fo J->I)."
    bool crossMatching;
    std::unique_ptr<IImageCollectionMatcher> imageCollectionMatcher = createImageCollectionMatcher(collectionMatcherType, distRatio, crossMatching);

    const std::vector<feature::EImageDescriberType> imageDescriberTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);

    LOG_INFO("There are %lu views and %lu image pairs.", m_sfmData->getViews().size(), pairs.size());

    LOG_INFO("Load features and descriptors");

    using namespace feature;
    // load the corresponding view regions
    RegionsPerView regionsPerView;
    
    std::atomic_bool invalid(false);
#pragma omp parallel num_threads(3)
 for(auto iter = m_sfmData->getViews().begin(); iter != m_sfmData->getViews().end() && !invalid; ++iter)
 {
#pragma omp single nowait
   {
     for(std::size_t i = 0; i < imageDescriberTypes.size(); ++i)
     {
       if(filter.empty() || filter.find(iter->second.get()->getViewId()) != filter.end())
       {
           std::unique_ptr<feature::Regions> regionsPtr = std::move(m_extractor->getRegionsList()[i]);
         if(regionsPtr)
         {
#pragma omp critical
           {
             regionsPerView.addRegions(iter->second.get()->getViewId(), imageDescriberTypes.at(i), regionsPtr.release());
//             ++progressDisplay;
           }
         }
         else
         {
           invalid = true;
         }
       }
     }
   }
 }
    
    // perform the matching
      system2::Timer timer;
      PairSet pairsPoseKnown;
      PairSet pairsPoseUnknown;

    
//    "Enable the usage of geometric information from known camera poses to guide the feature matching. "
//          "If some cameras have unknown poses (so there is no geometric prior), the standard feature matching will be performed."
    bool matchFromKnownCameraPoses = false;
      if(matchFromKnownCameraPoses)
      {
          for(const auto& p: pairs)
          {
            if(m_sfmData->isPoseAndIntrinsicDefined(p.first) && m_sfmData->isPoseAndIntrinsicDefined(p.second))
            {
                pairsPoseKnown.insert(p);
            }
            else
            {
                pairsPoseUnknown.insert(p);
            }
          }
      }
      else
      {
          pairsPoseUnknown = pairs;
      }
    
    if(!pairsPoseKnown.empty())
      {
        // compute matches from known camera poses when you have an initialization on the camera poses
        LOG_INFO("Putative matches from known poses: %lu image pairs.",pairsPoseKnown.size());

        sfm::StructureEstimationFromKnownPoses structureEstimator;
        
        structureEstimator.match(*m_sfmData, pairsPoseKnown, regionsPerView, knownPosesGeometricErrorMax);
        mapPutativesMatches = structureEstimator.getPutativesMatches();
      }

      if(!pairsPoseUnknown.empty())
      {
          LOG_INFO("Putative matches (unknown poses): %lu image pairs.", pairsPoseUnknown.size());
          // match feature descriptors between them without geometric notion
          
          int randomSeed = std::mt19937::default_seed;
          std::mt19937 randomNumberGenerator(randomSeed == -1 ? std::random_device()() : randomSeed);
          
          for(const feature::EImageDescriberType descType : imageDescriberTypes)
          {
            assert(descType != feature::EImageDescriberType::UNINITIALIZED);
            LOG_INFO("%s Regions Matching", EImageDescriberType_enumToString(descType).c_str());

            // photometric matching of putative pairs
            imageCollectionMatcher->Match(randomNumberGenerator, regionsPerView, pairsPoseUnknown, descType, mapPutativesMatches);

            // TODO: DELI
            // if(!guided_matching) regionPerView.clearDescriptors()
          }

      }

//    "A match is invalid if the 2d motion between the 2 points is less than a threshold (or -1 to disable this filter)."
      double minRequired2DMotion = -1.0;
    
      filterMatchesByMin2DMotion(mapPutativesMatches, regionsPerView, minRequired2DMotion);

      if(mapPutativesMatches.empty())
      {
        LOG_INFO("No putative feature matches.");
        // If we only compute a selection of matches, we may have no match.
        return !m_sfmData->views.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    
    std::string geometricFilterTypeName = matchingImageCollection::EGeometricFilterType_enumToString(matchingImageCollection::EGeometricFilterType::FUNDAMENTAL_MATRIX);
    const matchingImageCollection::EGeometricFilterType geometricFilterType = matchingImageCollection::EGeometricFilterType_stringToEnum(geometricFilterTypeName);
    
    using namespace matchingImageCollection;
      if(geometricFilterType == EGeometricFilterType::HOMOGRAPHY_GROWING)
      {
        // sort putative matches according to their Lowe ratio
        // This is suggested by [F.Srajer, 2016]: the matches used to be the seeds of the homographies growing are chosen according
        // to the putative matches order. This modification should improve recall.
        for(auto& imgPair: mapPutativesMatches)
        {
          for(auto& descType: imgPair.second)
          {
              matching::IndMatches & matches = descType.second;
            sortMatches_byDistanceRatio(matches);
          }
        }
      }

      // when a range is specified, generate a file prefix to reflect the current iteration (rangeStart/rangeSize)
      // => with matchFilePerImage: avoids overwriting files if a view is present in several iterations
      // => without matchFilePerImage: avoids overwriting the unique resulting file
      const std::string filePrefix = rangeSize > 0 ? std::to_string(rangeStart/rangeSize) + "." : "";

      LOG_INFO("%s putative image pair matches", std::to_string(mapPutativesMatches.size()).c_str());

      for(const auto& imageMatch: mapPutativesMatches)
          LOG_INFO("\t- image pair (%s, %s) contains %s putative matches.",std::to_string(imageMatch.first.first).c_str(),
                   std::to_string(imageMatch.first.second).c_str(),
                   std::to_string(imageMatch.second.getNbAllMatches()).c_str());
//        LOG_INFO("\t- image pair (" + std::to_string(imageMatch.first.first) << ", " + std::to_string(imageMatch.first.second) + ") contains " + std::to_string(imageMatch.second.getNbAllMatches()) + " putative matches.");

      // export putative matches
//      if(savePutativeMatches)
//        Save(mapPutativesMatches, (fs::path(matchesFolder) / "putativeMatches").string(), fileExtension, matchFilePerImage, filePrefix);

      LOG_INFO("Task (Regions Matching) done in (s): %.2f", timer.elapsed());

#ifdef ALICEVISION_DEBUG_MATCHING
    {
      LOG_DEBUG("PUTATIVE");
      getStatsMap(mapPutativesMatches);
    }
#endif
    
    // c. Geometric filtering of putative matches
      //    - AContrario Estimation of the desired geometric model
      //    - Use an upper bound for the a contrario estimated threshold

      timer.reset();
      

      matching::PairwiseMatches geometricMatches;

      LOG_INFO("Geometric filtering: using %s", matchingImageCollection::EGeometricFilterType_enumToString(geometricFilterType).c_str());

    //    "Maximum error (in pixels) allowed for features matching during geometric verification. "
    //          "If set to 0 it lets the ACRansac select an optimal value."
    double geometricErrorMax = 0.0; //< the maximum reprojection error allowed for image matching with geometric validation
    
    //Maximum number of iterations allowed in ransac step.
    int maxIteration = 2048;
    
    robustEstimation::ERobustEstimator geometricEstimator = robustEstimation::ERobustEstimator::ACRANSAC;
    
      switch(geometricFilterType)
      {

        case EGeometricFilterType::NO_FILTERING:
          geometricMatches = mapPutativesMatches;
        break;

        case EGeometricFilterType::FUNDAMENTAL_MATRIX:
        {
          matchingImageCollection::robustModelEstimation(geometricMatches,
            m_sfmData,
            regionsPerView,
            GeometricFilterMatrix_F_AC(geometricErrorMax, maxIteration, geometricEstimator),
            mapPutativesMatches,
            randomNumberGenerator,
            guidedMatching);
        }
        break;

//      case EGeometricFilterType::FUNDAMENTAL_WITH_DISTORTION:
//      {
//        matchingImageCollection::robustModelEstimation(geometricMatches,
//          &sfmData,
//          regionPerView,
//          GeometricFilterMatrix_F_AC(geometricErrorMax, maxIteration, geometricEstimator, true),
//          mapPutativesMatches,
//          randomNumberGenerator,
//          guidedMatching);
//      }
//      break;
//
//        case EGeometricFilterType::ESSENTIAL_MATRIX:
//        {
//          matchingImageCollection::robustModelEstimation(geometricMatches,
//            &sfmData,
//            regionPerView,
//            GeometricFilterMatrix_E_AC(geometricErrorMax, maxIteration),
//            mapPutativesMatches,
//            randomNumberGenerator,
//            guidedMatching);
//
//          removePoorlyOverlappingImagePairs(geometricMatches, mapPutativesMatches, 0.3f, 50);
//        }
//        break;
//
//        case EGeometricFilterType::HOMOGRAPHY_MATRIX:
//        {
//          const bool onlyGuidedMatching = true;
//          matchingImageCollection::robustModelEstimation(geometricMatches,
//            &sfmData,
//            regionPerView,
//            GeometricFilterMatrix_H_AC(geometricErrorMax, maxIteration),
//            mapPutativesMatches, randomNumberGenerator, guidedMatching,
//            onlyGuidedMatching ? -1.0 : 0.6);
//        }
//        break;
//
//        case EGeometricFilterType::HOMOGRAPHY_GROWING:
//        {
//          matchingImageCollection::robustModelEstimation(geometricMatches,
//            &sfmData,
//            regionPerView,
//            GeometricFilterMatrix_HGrowing(geometricErrorMax, maxIteration),
//            mapPutativesMatches,
//            randomNumberGenerator,
//            guidedMatching);
//        }
//        break;
      }

      ALICEVISION_LOG_INFO(std::to_string(geometricMatches.size()) + " geometric image pair matches:");
      for(const auto& matchGeo: geometricMatches)
        ALICEVISION_LOG_INFO("\t- image pair (" + std::to_string(matchGeo.first.first) + ", " + std::to_string(matchGeo.first.second) + ") contains " + std::to_string(matchGeo.second.getNbAllMatches()) + " geometric matches.");

      // grid filtering
      ALICEVISION_LOG_INFO("Grid filtering");

      PairwiseMatches finalMatches;
      matchesGridFilteringForAllPairs(geometricMatches, sfmData, regionPerView, useGridSort,
                                      numMatchesToKeep, finalMatches);

        ALICEVISION_LOG_INFO("After grid filtering:");
        for (const auto& matchGridFiltering: finalMatches)
        {
            ALICEVISION_LOG_INFO("\t- image pair (" << matchGridFiltering.first.first << ", "
                                 << matchGridFiltering.first.second << ") contains "
                                 << matchGridFiltering.second.getNbAllMatches()
                                 << " geometric matches.");
        }

      // export geometric filtered matches
      ALICEVISION_LOG_INFO("Save geometric matches.");
      Save(finalMatches, matchesFolder, fileExtension, matchFilePerImage, filePrefix);
      ALICEVISION_LOG_INFO("Task done in (s): " + std::to_string(timer.elapsed()));

    

    
    return true;
}
