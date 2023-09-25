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
#include <feature/ImageDescriber.hpp>

#include <system/Timer.hpp>
#include <SoftVisionLog.h>

#include <imageMatching/ImageMatching.hpp>
#include "voctree/descriptorLoader.hpp"

#include <matching/IndMatch.hpp>
#include <matching/matcherType.hpp>
#include <matching/matchesFiltering.hpp>
#include <matching/io.hpp>

#include <matchingImageCollection/IImageCollectionMatcher.hpp>
#include <matchingImageCollection/matchingCommon.hpp>
#include <matchingImageCollection/GeometricFilterType.hpp>
#include <matchingImageCollection/GeometricFilter.hpp>
#include <matchingImageCollection/GeometricFilterMatrix_F_AC.hpp>

#include <sfm/pipeline/structureFromKnownPoses/StructureEstimationFromKnownPoses.hpp>
#include <sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.hpp>

#include <robustEstimation/estimators.hpp>

#include <sfmDataIO/sfmDataIO.hpp>
#include <sfm/pipeline/regionsIO.hpp>
#include <sys/stat.h>

#include <softvision_omp.hpp>

#include <utils/strUtils.hpp>
#include <utils/fileUtil.hpp>

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
//  for(const auto& stat: stats)
//  {
//    LOG_DEBUG("%s\t%s", std::to_string(stat.first).c_str(), std::to_string(stat.second).c_str());
//  }
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
    delete mp_hwc;
}

void ReconPipeline::CameraInit(void) {
    printf("%p Do camera init ...\n", this);
    
    float fov =  45.0f; //degree
    float sensorWidthmm = 36.0f;
    //Construct camera intrinsic ...
    printf("fov == %.2f, sensorWidthmm == %.2f ...\n",fov,sensorWidthmm);
    
    m_sfmData = new sfmData::SfMData();
    mp_hwc = new HardwareContext();
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
        auto&& folder_name = m_outputFolder;
        std::string testimg_file_name = folder_name + std::to_string(viewId) + ".png";
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
//    views.insert(std::make_pair(IndexT(views.size()), pView));
    views.insert(std::make_pair(viewId, pView));
    
    float fov = PI / 4; //radian
    float sensorWidthmm = 36.0f;
    float focalRatio = 1.0f / (2.0f * tanf(fov / 2));
    float focalLength = sensorWidthmm * focalRatio;
    float fx = (focalLength / sensorWidthmm) * std::max(width_new, height_new);
    float fy = fx / focalRatio;

    //TODO:
    auto pIntrinsic = std::make_shared<camera::PinholeRadialK3>(width_new, height_new, fx, fy);
    
    auto&& intrinsics = m_sfmData->intrinsics;
    intrinsics.insert(std::make_pair(intrinsicId, pIntrinsic));
    
    {
        using namespace sfmDataIO;
        m_sfmData->setAbsolutePath(m_outputFolder + "cameraInit.sfm");
        Save(*m_sfmData, m_outputFolder, "cameraInit.sfm", ESfMData(VIEWS|INTRINSICS));
    }
    
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
 
    using namespace sfmDataIO;
    sfmData::SfMData tmpData;
    // If cached data exists then use it.
    if(m_sfmData->views.empty() && Load(tmpData, m_outputFolder, "cameraInit.sfm", ESfMData(VIEWS|INTRINSICS)))
    {
        *m_sfmData = tmpData;
    }
    
    m_extractor = new featureEngine::FeatureExtractor(*m_sfmData);

    auto&& extractor = *m_extractor;
    
    int rangeStart = 0, rangeSize = m_sfmData->views.size();
    extractor.setRange(rangeStart, rangeSize);
    
    //TODO: performance optimization
//    m_describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::DSPSIFT);
    
    auto descType = feature::EImageDescriberType::SIFT;
    m_describerTypesName = feature::EImageDescriberType_enumToString(descType);
    
    bool needExtract = false;
    for(auto&& item : m_sfmData->views)
    {
        auto&& prefix = m_outputFolder + std::to_string(item.first) + "." + m_describerTypesName;

        struct stat buffer1;
        struct stat buffer2;
        if((stat((prefix + ".desc").c_str(), &buffer1)) != 0 ||
           (stat((prefix + ".feat").c_str(), &buffer2)) != 0) {
            LOG_INFO("feat/desc files are not complete, need do extract.");
            needExtract = true;
            break;
        }
    }
    if(!needExtract) {
        //loading
        
        auto&& mpRegions = m_extractor->getRegionsPerView();
        for(auto&& item : m_sfmData->views)
        {
            auto&& prefix = m_outputFolder + std::to_string(item.first) + "." + m_describerTypesName;
            
            std::unique_ptr<feature::Regions> regions;
            
            std::unique_ptr<feature::ImageDescriber> decriber = feature::createImageDescriber(descType);
            (*decriber).allocate(regions);
            
            regions->Load(prefix + ".feat", prefix + ".desc");
            mpRegions.addRegions(item.first, descType, regions.release());
        }
        
        return true;
    }
    
//    std::string names;
//    for(int i = 0; i < rangeSize; ++i)
//    {
//        names += m_describerTypesName + ",";
//    }
//    m_describerTypesName = names;
    
    {
        std::vector<feature::EImageDescriberType> imageDescriberTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);

        for(const auto& imageDescriberType: imageDescriberTypes)
        {
            std::shared_ptr<feature::ImageDescriber> imageDescriber = feature::createImageDescriber(imageDescriberType);
            
            feature::ConfigurationPreset featDescConfig;
            featDescConfig.setDescPreset(feature::EImageDescriberPreset::NORMAL);
            featDescConfig.setContrastFiltering(feature::EFeatureConstrastFiltering::GridSort);
            featDescConfig.setGridFiltering(true);
            
            imageDescriber->setConfigurationPreset(featDescConfig);
            extractor.addImageDescriber(imageDescriber);
        }
    }
    
    // set maxThreads
//    HardwareContext hwc;
    if(!mp_hwc) {
        LOG_ERROR("HardwareContext is NULL!!");
        return false;
    }
    mp_hwc->setUserCoresLimit(8);
    
    // feature extraction routines
    // for each View of the SfMData container:
    // - if regions file exist continue,
    // - if no file, compute features
    {
        system2::Timer timer;
        
        m_featureFolder = m_outputFolder + "features/";
        if(!utils::create_directory(m_featureFolder)) {
            LOG_ERROR("Create directory %s Failed!", m_featureFolder.c_str());
        }
        extractor.setOutputFolder(m_featureFolder);
        extractor.process(*mp_hwc, image::EImageColorSpace::SRGB);

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
    std::size_t numImageQuery = 40;
    /// the number of neighbors to retrieve for each image in Sequential Mode
    std::size_t numImageQuerySequential = 5;

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
    float distRatio = 0.8f;
//    "Make sure that the matching process is symmetric (same matches for I->J than fo J->I)."
    bool crossMatching = false;
    std::unique_ptr<IImageCollectionMatcher> imageCollectionMatcher = createImageCollectionMatcher(collectionMatcherType, distRatio, crossMatching);
    
    const std::vector<feature::EImageDescriberType> imageDescriberTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);

    LOG_INFO("There are %lu views and %lu image pairs.", m_sfmData->getViews().size(), pairs.size());

    LOG_INFO("Load features and descriptors");

    using namespace feature;
    // load the corresponding view regions
    RegionsPerView regionsPerView;
    
    if(m_extractor->getRegionsPerView().isEmpty()) {
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
    //           std::unique_ptr<feature::Regions> regionsPtr = std::move(m_extractor->getRegionsPerView()[(*iter)->v]);
               auto* regionsPtr = (feature::Regions*)&(m_extractor->getRegionsPerView().getRegions(iter->second.get()->getViewId(), imageDescriberTypes[i]));
    //           m_extractor->getRegionsPerView()
             if(regionsPtr)
             {
    #pragma omp critical
               {
                   LOG_DEBUG("regionsPerView.addRegions viewid=%u regionsptr==%p", iter->second.get()->getViewId(), regionsPtr);
                 regionsPerView.addRegions(iter->second.get()->getViewId(), imageDescriberTypes.at(i), regionsPtr);
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
    }
    else {
        regionsPerView = std::move(m_extractor->getRegionsPerView());
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
    //          "If set to 0 it lets the ACRansac select an optimal value. ??"
    //TODO: check to set a reasonable value!
    double geometricErrorMax = 4.0; //< the maximum reprojection error allowed for image matching with geometric validation
    
    //Maximum number of iterations allowed in ransac step.
    int maxIteration = 2048;
    
    //"This seed value will generate a sequence using a linear random generator. Set -1 to use a random seed."
    int randomSeed = std::mt19937::default_seed;
    std::mt19937 randomNumberGenerator(randomSeed == -1 ? std::random_device()() : randomSeed);
    
    //Use the found model to improve the pairwise correspondences.
    bool guidedMatching = false;
    
    robustEstimation::ERobustEstimator geometricEstimator = robustEstimation::ERobustEstimator::ACRANSAC;
    
      switch(geometricFilterType)
      {

        case EGeometricFilterType::NO_FILTERING:
          geometricMatches = mapPutativesMatches;
        break;

        case EGeometricFilterType::FUNDAMENTAL_MATRIX:
        {
          LOG_INFO("case EGeometricFilterType::FUNDAMENTAL_MATRIX");
          matchingImageCollection::robustModelEstimation(geometricMatches,
            (const sfmData::SfMData*)m_sfmData,
            regionsPerView,
            GeometricFilterMatrix_F_AC(geometricErrorMax, maxIteration, geometricEstimator),
            mapPutativesMatches,
            randomNumberGenerator,
            guidedMatching);
        }
        break;
      }

      LOG_INFO("%lu geometric image pair matches:", geometricMatches.size());
      for(const auto& matchGeo: geometricMatches)
          LOG_INFO("\t- image pair (%u,%u) contains %d geometric matches.", matchGeo.first.first, matchGeo.first.second, matchGeo.second.getNbAllMatches());

      // grid filtering
      LOG_INFO("Grid filtering");

    //Use matching grid sort
      bool useGridSort = true;
    //Maximum number of matches to keep
    size_t numMatchesToKeep = 0;
    
      PairwiseMatches finalMatches;
      matchesGridFilteringForAllPairs(geometricMatches, *m_sfmData, regionsPerView, useGridSort,
                                      numMatchesToKeep, finalMatches);

        LOG_INFO("After grid filtering:");
        for (const auto& matchGridFiltering: finalMatches)
        {
            LOG_INFO("\t- image pair (%u,%u) contains %d geometric matches.", matchGridFiltering.first.first, matchGridFiltering.first.second, matchGridFiltering.second.getNbAllMatches());
        }

      // export geometric filtered matches
      LOG_INFO("Save geometric matches.");
    bool matchFilePerImage = false;
    const std::string fileExtension = "txt";
    m_matchesFolder = m_outputFolder + "matches/";
    if(!utils::create_directory(m_matchesFolder))
        LOG_ERROR("Create matches dir FAILED!");
      Save(finalMatches, m_matchesFolder, fileExtension, matchFilePerImage, filePrefix);
      LOG_INFO("Task done in (s): %.2f",timer.elapsed());

    return true;
}

/**
 * @brief Retrieve the view id in the sfmData from the image filename.
 * @param[in] sfmData the SfM scene
 * @param[in] name the image name to find (uid or filename or path)
 * @param[out] out_viewId the id found
 * @return if a view is found
 */
bool retrieveViewIdFromImageName(const sfmData::SfMData& sfmData,
                                 const std::string& name,
                                 IndexT& out_viewId)
{
  out_viewId = UndefinedIndexT;

  // list views uid / filenames and find the one that correspond to the user ones
  for(const auto& viewPair : sfmData.getViews())
  {
    const sfmData::View& v = *(viewPair.second.get());
    
    if(name == std::to_string(v.getViewId()) ||
       name == utils::GetFileName(v.getImagePath()) ||
       name == v.getImagePath())
    {
      out_viewId = v.getViewId();
      break;
    }
  }

  if(out_viewId == UndefinedIndexT)
    LOG_X("Can't find the given initial pair view: " << name);

  return out_viewId != UndefinedIndexT;
}

int ReconPipeline::IncrementalSFM()
{
    LOG_INFO("StructureFromMotion\n"
             "- This program performs incremental SfM (Initial Pair Essential + Resection)\n");
    
    omp_set_num_threads(mp_hwc->getMaxThreads());

    sfm::ReconstructionEngine_sequentialSfM::Params sfmParams; //TODO: set it properly
    
    const double defaultLoRansacLocalizationError = 4.0;
    if(!robustEstimation::adjustRobustEstimatorThreshold(sfmParams.localizerEstimator, sfmParams.localizerEstimatorError, defaultLoRansacLocalizationError))
    {
        return EXIT_FAILURE;
    }
    
    // load input SfMData scene
    sfmData::SfMData sfmData;
    if(sfmDataIO::Load(sfmData, m_outputFolder, "cameraInit.sfm", sfmDataIO::ESfMData::ALL))
    {
        *m_sfmData = sfmData;
    }
    
    bool lockScenePreviouslyReconstructed = true;
    // lock scene previously reconstructed
      if(lockScenePreviouslyReconstructed)
      {
        // lock all reconstructed camera poses
        for(auto& cameraPosePair : sfmData.getPoses())
          cameraPosePair.second.lock();

        for(const auto& viewPair : sfmData.getViews())
        {
          // lock all reconstructed views intrinsics
          const sfmData::View& view = *(viewPair.second);
          if(sfmData.isPoseAndIntrinsicDefined(&view))
            sfmData.getIntrinsics().at(view.getIntrinsicId())->lock();
        }
      }
    
    // get imageDescriber type
      const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);

      // features reading
      feature::FeaturesPerView featuresPerView;
    
      std::vector<std::string> featuresFolders{m_featureFolder};
    
      if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
      {
        LOG_ERROR("Invalid features.");
        return EXIT_FAILURE;
      }
    
    
    // matches reading
    matching::PairwiseMatches pairwiseMatches;
    std::vector<std::string> matchesFolders{m_matchesFolder};
    int maxNbMatches = 0;
    int minNbMatches = 0;
    bool useOnlyMatchesFromInputFolder = false;
    if(!sfm::loadPairwiseMatches(pairwiseMatches, sfmData, matchesFolders, describerTypes, maxNbMatches, minNbMatches, useOnlyMatchesFromInputFolder))
    {
      LOG_ERROR("Unable to load matches.");
      return EXIT_FAILURE;
    }
    
    // sequential reconstruction process
      system2::Timer timer;

      if(sfmParams.minNbObservationsForTriangulation < 2)
      {
        // allows to use to the old triangulatation algorithm (using 2 views only) during resection.
        sfmParams.minNbObservationsForTriangulation = 0;
        // LOG_ERROR("The value associated to the argument '--minNbObservationsForTriangulation' must be >= 2 ");
        // return EXIT_FAILURE;
      }

    std::pair<std::string,std::string> initialPairString("","");
      // handle initial pair parameter
      if(!initialPairString.first.empty() || !initialPairString.second.empty())
      {
        if(initialPairString.first == initialPairString.second)
        {
          LOG_ERROR("Invalid image names. You cannot use the same image to initialize a pair.");
          return EXIT_FAILURE;
        }

        if(!initialPairString.first.empty() && !retrieveViewIdFromImageName(sfmData, initialPairString.first, sfmParams.userInitialImagePair.first))
        {
          LOG_X("Could not find corresponding view in the initial pair: " + initialPairString.first);
          return EXIT_FAILURE;
        }

        if(!initialPairString.second.empty() && !retrieveViewIdFromImageName(sfmData, initialPairString.second, sfmParams.userInitialImagePair.second))
        {
            LOG_X("Could not find corresponding view in the initial pair: " + initialPairString.second);
          return EXIT_FAILURE;
        }
      }
    
    std::string extraInfoFolder = m_outputFolder + "StructureFromMotion/";
    if(!utils::exists(extraInfoFolder)) {
        utils::create_directory(extraInfoFolder);
    }
    sfm::ReconstructionEngine_sequentialSfM sfmEngine(
        sfmData,
        sfmParams,
        extraInfoFolder,
        (extraInfoFolder +  "sfm_log.html"));

    int randomSeed = std::mt19937::default_seed;
    sfmEngine.initRandomSeed(randomSeed);

      // configure the featuresPerView & the matches_provider
      sfmEngine.setFeatures(&featuresPerView);
      sfmEngine.setMatches(&pairwiseMatches);

      if(!sfmEngine.process())
        return EXIT_FAILURE;

      // set featuresFolders and matchesFolders relative paths
      {
          sfmEngine.getSfMData().addFeaturesFolders(featuresFolders);
          sfmEngine.getSfMData().addMatchesFolders(matchesFolders);
          sfmEngine.getSfMData().setAbsolutePath(m_outputFolder + "sfm.abc");
      }

    bool computeStructureColor = true;
    // get the color for the 3D points
    if(computeStructureColor)
      sfmEngine.colorize();
    
    sfmEngine.retrieveMarkersId();

    LOG_X("Structure from motion took (s): " + std::to_string(timer.elapsed()));
    
    return EXIT_SUCCESS;
}
