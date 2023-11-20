//
//  camerainit.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/13.
//

#include <ReconPipeline.h>

#include <utils/YuvImageProcessor.h>
#include "PngUtils.h"

#include <sfmData/SfMData.hpp>
#include <sfmData/View.hpp>
#include <camera/PinholeRadial.hpp>
#include "camera/cameraUndistortImage.hpp"
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
#include <sfm/generateReport.hpp>

#include <sys/stat.h>

#include <softvision_omp.hpp>

#include <utils/strUtils.hpp>
#include <utils/fileUtil.hpp>
#include <utils/PngUtils.h>

#include <image/io.hpp>
#include <image/convertion.hpp>

#include <mvsUtils/MultiViewParams.hpp>
#include <depthMap/computeOnMultiGPUs.hpp>
#include <depthMap/DepthMapEstimator.hpp>
#include <depthMap/DepthMapParams.hpp>
#include <depthMap/SgmParams.hpp>
#include <depthMap/RefineParams.hpp>
#include <gpu/gpu.hpp>

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
    m_matchesFolder = m_outputFolder + "matches/";
    m_featureFolder = m_outputFolder + "features/";
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
        auto&& prefix = m_featureFolder + std::to_string(item.first) + "." + m_describerTypesName;
        
        if(!utils::exists(prefix + ".desc") || !utils::exists(prefix + ".feat")) {
            needExtract = true;
            break;
        }
    }
    if(!needExtract) {
        //loading
        
        auto&& mpRegions = m_extractor->getRegionsPerView();
        for(auto&& item : m_sfmData->views)
        {
            auto&& prefix = m_featureFolder + std::to_string(item.first) + "." + m_describerTypesName;
            
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
    
    // If exists then load matches
    matching::PairwiseMatches pairwiseMatches;
    std::vector<std::string> matchesFolders{m_matchesFolder};
    int maxNbMatches = 0;
    int minNbMatches = 0;
    bool useOnlyMatchesFromInputFolder = false;
    const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(m_describerTypesName);
    if(sfm::loadPairwiseMatches(pairwiseMatches, *m_sfmData, matchesFolders, describerTypes, maxNbMatches, minNbMatches, useOnlyMatchesFromInputFolder))
    {
      LOG_INFO("Load cached matches.");
      return EXIT_SUCCESS;
    }
    
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
    
    std::string extraInfoFolder = m_outputFolder + "StructureFromMotion/";
    std::string const& outputSfM = "sfm.abc";
    
    sfmData::SfMData tmpData;
    if(sfmDataIO::Load(tmpData, extraInfoFolder, extraInfoFolder + outputSfM, sfmDataIO::ESfMData::ALL)) {
        LOG_INFO("IncrementalSFM already have done! Now use cached data ...");
//        *m_sfmData = tmpData;
        m_sfmData->structure = tmpData.structure;
        m_sfmData->setAbsolutePath(tmpData.getAbsolutePath());
        m_sfmData->setFeaturesFolders(tmpData.getFeaturesFolders());
        m_sfmData->setMatchesFolders(tmpData.getMatchesFolders());
        m_sfmData->getPoses() = tmpData.getPoses();
        
        return EXIT_SUCCESS;
    }
    
    
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
    
    
    if(!utils::exists(extraInfoFolder)) {
        utils::create_directory(extraInfoFolder);
    }
    
    sfmParams.sfmStepFileExtension  = ".abc";
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
    
    LOG_INFO("Generating HTML report...");

    sfm::generateSfMReport(sfmEngine.getSfMData(), extraInfoFolder + "sfm_report.html");

    // export to disk computed scene (data & visualizable results)
    
    LOG_X("Export SfMData to disk: " + outputSfM);

    
    sfmDataIO::Save(sfmEngine.getSfMData(), extraInfoFolder, "cloud_and_poses" + sfmParams.sfmStepFileExtension, sfmDataIO::ESfMData(sfmDataIO::VIEWS|sfmDataIO::EXTRINSICS|sfmDataIO::INTRINSICS|sfmDataIO::STRUCTURE));
    sfmDataIO::Save(sfmEngine.getSfMData(), extraInfoFolder,outputSfM, sfmDataIO::ESfMData::ALL);

    std::string outputSfMViewsAndPoses = "cloud_and_poses.abc"; //"Path to the output SfMData file (with only views and poses)."
    if(!outputSfMViewsAndPoses.empty())
     sfmDataIO:: Save(sfmEngine.getSfMData(), extraInfoFolder, outputSfMViewsAndPoses, sfmDataIO::ESfMData(sfmDataIO::VIEWS|sfmDataIO::EXTRINSICS|sfmDataIO::INTRINSICS));

    LOG_X("Structure from Motion results:" << std::endl
      << "\t- # input images: " << sfmEngine.getSfMData().getViews().size() << std::endl
      << "\t- # cameras calibrated: " << sfmEngine.getSfMData().getValidViews().size() << std::endl
      << "\t- # poses: " << sfmEngine.getSfMData().getPoses().size() << std::endl
      << "\t- # landmarks: " << sfmEngine.getSfMData().getLandmarks().size());
    
    *m_sfmData = sfmEngine.getSfMData();

    return EXIT_SUCCESS;
}

template <class ImageT, class MaskFuncT>
void process(const std::string &dstColorImage, const camera::IntrinsicBase* cam, const oiio::ParamValueList & metadata, ImageT& image, bool evCorrection, float exposureCompensation, MaskFuncT && maskFunc)
{
//    ImageT image, image_ud;
    ImageT image_ud;
//    readImage(srcImage, image, image::EImageColorSpace::LINEAR);

    // exposure correction
    if(evCorrection)
    {
        for(int pix = 0; pix < image.Width() * image.Height(); ++pix)
        {
            image(pix) = image(pix) * exposureCompensation;
        }
    }

    // mask
    maskFunc(image);

    // undistort //TODO: write exr
    if(cam->isValid() && cam->hasDistortion())
    {
        // undistort the image and save it
        using Pix = typename ImageT::Tpixel;
        Pix pixZero(Pix::Zero());
        UndistortImage(image, cam, image_ud, pixZero);
        
        writeImage(dstColorImage, image_ud, image::ImageWriteOptions(), metadata);
        LOG_DEBUG("write iMage undistort");
    }
    else
    {
        writeImage(dstColorImage, image, image::ImageWriteOptions(), metadata);
        LOG_DEBUG("write iMage original");
    }
}

int ReconPipeline::PrepareDenseScene()
{
    // defined view Ids
    std::set<IndexT> viewIds;

    if(!m_sfmData) return false;
    
    auto&& sfmData = *m_sfmData;
    sfmData::Views::const_iterator itViewBegin = sfmData.getViews().begin();
    sfmData::Views::const_iterator itViewEnd = sfmData.getViews().end();

//    int endIndex = ?; int beginIndex = ?;
//    if(endIndex > 0)
//    {
//        itViewEnd = itViewBegin;
//        std::advance(itViewEnd, endIndex);
//    }

//    std::advance(itViewBegin, (beginIndex < 0) ? 0 : beginIndex);
    
    // export valid views as projective cameras
    for(auto it = itViewBegin; it != itViewEnd; ++it)
    {
        const sfmData::View* view = it->second.get();
        if (!sfmData.isPoseAndIntrinsicDefined(view))
            continue;
        viewIds.insert(view->getViewId());
    }

    image::EImageFileType outputFileType = image::EImageFileType::EXR;
    bool saveMetadata = true;
    
    if((outputFileType != image::EImageFileType::EXR) && saveMetadata)
        LOG_INFO("Cannot save informations in images metadata.\n"
                                "Choose '.exr' file type if you want custom metadata");

    // export data
    auto progressDisplay = system2::createConsoleProgressDisplay(viewIds.size(), std::cout,
                                                                "Exporting Scene Undistorted Images\n");

    // for exposure correction
    const double medianCameraExposure = sfmData.getMedianCameraExposureSetting().getExposure();
    LOG_X("Median Camera Exposure: " << medianCameraExposure << ", Median EV: " << std::log2(1.0/medianCameraExposure));
    
#pragma omp parallel for num_threads(3)
    for(int i = 0; i < viewIds.size(); ++i)
    {
        auto itView = viewIds.begin();
        std::advance(itView, i);

        const IndexT viewId = *itView;
        const sfmData::View* view = sfmData.getViews().at(viewId).get();

        sfmData::Intrinsics::const_iterator iterIntrinsic = sfmData.getIntrinsics().find(view->getIntrinsicId());

        // we have a valid view with a corresponding camera & pose
        const std::string baseFilename = std::to_string(viewId);

        const std::string dstColorImage = m_outputFolder + baseFilename + "." + image::EImageFileType_enumToString(outputFileType);
        if(utils::exists(dstColorImage)) {
            LOG_INFO("dstColorImage %s already exist!", dstColorImage.c_str());
            continue;
        }
        // get metadata from source image to be sure we get all metadata. We don't use the metadatas from the Views inside the SfMData to avoid type conversion problems with string maps.
        std::string srcImage = view->getImagePath();
        oiio::ParamValueList metadata;// = image::readImageMetadata(srcImage); //TODO: fill it

        bool saveMatricesFiles = false;
        // export camera
        if(saveMetadata || saveMatricesFiles)
        {
            // get camera pose / projection
            const geometry::Pose3 pose = sfmData.getPose(*view).getTransform();

            std::shared_ptr<camera::IntrinsicBase> cam = iterIntrinsic->second;
            std::shared_ptr<camera::Pinhole> camPinHole = std::dynamic_pointer_cast<camera::Pinhole>(cam);
            if (!camPinHole) {
                LOG_ERROR("Camera is not pinhole in filter");
                continue;
            }

            Mat34 P = camPinHole->getProjectiveEquivalent(pose);

            // get camera intrinsics matrices
            const Mat3 K = dynamic_cast<const camera::Pinhole*>(sfmData.getIntrinsicPtr(view->getIntrinsicId()))->K();
            const Mat3& R = pose.rotation();
            const Vec3& t = pose.translation();

            if(saveMatricesFiles)
            {
                std::ofstream fileP(m_outputFolder + baseFilename + "_P.txt");
                fileP << std::setprecision(10)
                        << P(0, 0) << " " << P(0, 1) << " " << P(0, 2) << " " << P(0, 3) << "\n"
                        << P(1, 0) << " " << P(1, 1) << " " << P(1, 2) << " " << P(1, 3) << "\n"
                        << P(2, 0) << " " << P(2, 1) << " " << P(2, 2) << " " << P(2, 3) << "\n";
                fileP.close();

                std::ofstream fileKRt(m_outputFolder + baseFilename + "_KRt.txt");
                fileKRt << std::setprecision(10)
                        << K(0, 0) << " " << K(0, 1) << " " << K(0, 2) << "\n"
                        << K(1, 0) << " " << K(1, 1) << " " << K(1, 2) << "\n"
                        << K(2, 0) << " " << K(2, 1) << " " << K(2, 2) << "\n"
                        << "\n"
                        << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << "\n"
                        << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << "\n"
                        << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << "\n"
                        << "\n"
                        << t(0) << " " << t(1) << " " << t(2) << "\n";
                fileKRt.close();
            }

            if(saveMetadata)
            {
                // convert to 44 matix
                Mat4 projectionMatrix;
                projectionMatrix << P(0, 0), P(0, 1), P(0, 2), P(0, 3),
                                    P(1, 0), P(1, 1), P(1, 2), P(1, 3),
                                    P(2, 0), P(2, 1), P(2, 2), P(2, 3),
                                         0,       0,       0,       1;

                // convert matrices to rowMajor
                std::vector<double> vP(projectionMatrix.size());
                std::vector<double> vK(K.size());
                std::vector<double> vR(R.size());

                typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;
                Eigen::Map<RowMatrixXd>(vP.data(), projectionMatrix.rows(), projectionMatrix.cols()) = projectionMatrix;
                Eigen::Map<RowMatrixXd>(vK.data(), K.rows(), K.cols()) = K;
                Eigen::Map<RowMatrixXd>(vR.data(), R.rows(), R.cols()) = R;

                // add metadata
                metadata.push_back(oiio::ParamValue("SoftVision:downscale", 1));
                metadata.push_back(oiio::ParamValue("SoftVision:P", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX44), 1, vP.data()));
                metadata.push_back(oiio::ParamValue("SoftVision:K", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX33), 1, vK.data()));
                metadata.push_back(oiio::ParamValue("SoftVision:R", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX33), 1, vR.data()));
                metadata.push_back(oiio::ParamValue("SoftVision:t", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::VEC3), 1, t.data()));
            }
        }

        // export undistort image
        {
//            if(!imagesFolders.empty())
//            {
//                std::vector<std::string> paths = sfmDataIO::viewPathsFromFolders(*view, imagesFolders);
//
//                // if path was not found
//                if(paths.empty())
//                {
//                    throw std::runtime_error("Cannot find view '" + std::to_string(view->getViewId()) + "' image file in given folder(s)");
//                }
//                else if(paths.size() > 1)
//                {
//                    throw std::runtime_error( "Ambiguous case: Multiple source image files found in given folder(s) for the view '" +
//                        std::to_string(view->getViewId()) + "'.");
//                }
//
//                srcImage = paths[0];
//            }
//            const std::string dstColorImage = (fs::path(outFolder) / (baseFilename + "." + image::EImageFileType_enumToString(outputFileType))).string();
            
            const camera::IntrinsicBase* cam = iterIntrinsic->second.get();

            // add exposure values to images metadata
            const double cameraExposure = view->getCameraExposureSetting().getExposure();
            const double ev = std::log2(1.0 / cameraExposure);
            const float exposureCompensation = float(medianCameraExposure / cameraExposure);
            metadata.push_back(oiio::ParamValue("SoftVision:EV", float(ev)));
            metadata.push_back(oiio::ParamValue("SoftVision:EVComp", exposureCompensation));

            bool evCorrection = false;
            if(evCorrection)
            {
                LOG_X("image " << viewId << ", exposure: " << cameraExposure << ", Ev " << ev << " Ev compensation: " + std::to_string(exposureCompensation));
            }

//            image::Image<unsigned char> mask;
//            if(tryLoadMask(&mask, masksFolders, viewId, srcImage))
//            {
//                process<Image<RGBAfColor>>(dstColorImage, cam, metadata, srcImage, evCorrection, exposureCompensation, [&mask] (Image<RGBAfColor> & image)
//                {
//                    if(image.Width() * image.Height() != mask.Width() * mask.Height())
//                    {
//                        LOG_X("Invalid image mask size: mask is ignored.");
//                        return;
//                    }
//
//                    for(int pix = 0; pix < image.Width() * image.Height(); ++pix)
//                    {
//                        const bool masked = (mask(pix) == 0);
//                        image(pix).a() = masked ? 0.f : 1.f;
//                    }
//                });
//            }
//            else
//            {
            image::Image<image::RGBAColor> imageRGBA;
            
            image::byteBuffer2EigenMatrix(view->getWidth(), view->getHeight(), view->getBuffer(), imageRGBA);
            
            const auto noMaskingFunc = [] (image::Image<image::RGBAColor> const& image) {};
                process<image::Image<image::RGBAColor>>(dstColorImage, cam, metadata, imageRGBA, evCorrection, exposureCompensation, noMaskingFunc);
//            }
        }

        ++progressDisplay;
    }

    LOG_INFO("PrepareDenseScene Done!");
    return EXIT_SUCCESS;
}

int computeDownscale(const mvsUtils::MultiViewParams& mp, int scale, int maxWidth, int maxHeight)
{
    const int maxImageWidth = mp.getMaxImageWidth() / scale;
    const int maxImageHeight = mp.getMaxImageHeight() / scale;

    int downscale = 1;
    int downscaleWidth = mp.getMaxImageWidth() / scale;
    int downscaleHeight = mp.getMaxImageHeight() / scale;

    while((downscaleWidth > maxWidth) || (downscaleHeight > maxHeight))
    {
        downscale++;
        downscaleWidth = maxImageWidth / downscale;
        downscaleHeight = maxImageHeight / downscale;
    }

    return downscale;
}

int ReconPipeline::DepthMapEstimation()
{
    LOG_INFO("Dense Reconstruction.\n"
             "This program estimate a depth map for each input calibrated camera using Plane Sweeping, a multi-view stereo algorithm notable for its efficiency on modern graphics hardware (GPU).\n"
             "SoftVision depthMapEstimation");
    
    // program range
    int rangeStart = -1;
    int rangeSize = -1;

    // global image downscale factor
    int downscale = 2;

    // min / max view angle
    float minViewAngle = 2.0f;
    float maxViewAngle = 70.0f;

    // Tiling parameters
    mvsUtils::TileParams tileParams;

    // DepthMap (global) parameters
    depthMap::DepthMapParams depthMapParams;

    // Semi Global Matching Parameters
    depthMap::SgmParams sgmParams;

    // Refine Parameters
    depthMap::RefineParams refineParams;

    // intermediate results
    bool exportIntermediateDepthSimMaps = false;
    bool exportIntermediateNormalMaps = false;
    bool exportIntermediateVolumes = false;
    bool exportIntermediateCrossVolumes = false;
    bool exportIntermediateTopographicCutVolumes = false;
    bool exportIntermediateVolume9pCsv = false;
    
    // intermediate results
    sgmParams.exportIntermediateDepthSimMaps = exportIntermediateDepthSimMaps;
    sgmParams.exportIntermediateNormalMaps = exportIntermediateNormalMaps;
    sgmParams.exportIntermediateVolumes = exportIntermediateVolumes;
    sgmParams.exportIntermediateCrossVolumes = exportIntermediateCrossVolumes;
    sgmParams.exportIntermediateTopographicCutVolumes = exportIntermediateTopographicCutVolumes;
    sgmParams.exportIntermediateVolume9pCsv = exportIntermediateVolume9pCsv;

    refineParams.exportIntermediateDepthSimMaps = exportIntermediateDepthSimMaps;
    refineParams.exportIntermediateNormalMaps = exportIntermediateNormalMaps;
    refineParams.exportIntermediateCrossVolumes = exportIntermediateCrossVolumes;
    refineParams.exportIntermediateTopographicCutVolumes = exportIntermediateTopographicCutVolumes;
    refineParams.exportIntermediateVolume9pCsv = exportIntermediateVolume9pCsv;
    
    gpu::gpuInformation();
    
    // check if the scale is correct
    if(downscale < 1)
    {
      LOG_ERROR("Invalid value for downscale parameter. Should be at least 1.");
      return EXIT_FAILURE;
    }
    
    // check that Sgm scaleStep is greater or equal to the Refine scaleStep
    if(depthMapParams.useRefine)
    {
      const int sgmScaleStep = sgmParams.scale * sgmParams.stepXY;
      const int refineScaleStep = refineParams.scale * refineParams.stepXY;

      if(sgmScaleStep < refineScaleStep)
      {
        LOG_ERROR("SGM downscale (scale x step) should be greater or equal to the Refine downscale (scale x step).");
        return EXIT_FAILURE;
      }

      if(sgmScaleStep % refineScaleStep != 0)
      {
        LOG_ERROR("SGM downscale (scale x step) should be a multiple of the Refine downscale (scale x step).");
        return EXIT_FAILURE;
      }
    }
    
    // check min/max view angle
    if(minViewAngle < 0.f || minViewAngle > 360.f ||
       maxViewAngle < 0.f || maxViewAngle > 360.f ||
       minViewAngle > maxViewAngle)
    {
      LOG_ERROR("Invalid value for minViewAngle/maxViewAngle parameter(s). Should be between 0 and 360.");
      return EXIT_FAILURE;
    }
    
    // MultiViewParams initialization
    auto& imagesFolder = m_outputFolder;
    mvsUtils::MultiViewParams mp(*m_sfmData, imagesFolder, m_outputFolder, "", false, downscale);
    
    // set MultiViewParams min/max view angle
    mp.setMinViewAngle(minViewAngle);
    mp.setMaxViewAngle(maxViewAngle);

    // set undefined tile dimensions
    if(tileParams.bufferWidth <= 0 || tileParams.bufferHeight <= 0)
    {
      tileParams.bufferWidth  = mp.getMaxImageWidth();
      tileParams.bufferHeight = mp.getMaxImageHeight();
    }

    // check if the tile padding is correct
    if(tileParams.padding < 0 &&
       tileParams.padding * 2 < tileParams.bufferWidth &&
       tileParams.padding * 2 < tileParams.bufferHeight)
    {
        LOG_ERROR("Invalid value for tilePadding parameter. Should be at least 0 and not exceed half buffer width and height.");
        return EXIT_FAILURE;
    }
    
    // check if tile size > max image size
    if(tileParams.bufferWidth > mp.getMaxImageWidth() || tileParams.bufferHeight > mp.getMaxImageHeight())
    {
        LOG_X("Tile buffer size (width: "  << tileParams.bufferWidth << ", height: " << tileParams.bufferHeight
                                << ") is larger than the maximum image size (width: " << mp.getMaxImageWidth() << ", height: " << mp.getMaxImageHeight() <<  ").");
    }
    
    // check if SGM scale and step are set to -1
    bool autoSgmScaleStep = false;

    // compute SGM scale and step
    if(sgmParams.scale == -1 || sgmParams.stepXY == -1)
    {
        const int fileScale = 1; // input images scale (should be one)
        const int maxSideXY = 700 / mp.getProcessDownscale(); // max side in order to fit in device memory
        const int maxImageW = mp.getMaxImageWidth();
        const int maxImageH = mp.getMaxImageHeight();

        int maxW = maxSideXY;
        int maxH = maxSideXY * 0.8;

        if(maxImageW < maxImageH)
            std::swap(maxW, maxH);

        if(sgmParams.scale == -1)
        {
            // compute the number of scales that will be used in the plane sweeping.
            // the highest scale should have a resolution close to 700x550 (or less).
            const int scaleTmp = computeDownscale(mp, fileScale, maxW, maxH);
            sgmParams.scale = std::min(2, scaleTmp);
        }

        if(sgmParams.stepXY == -1)
        {
            sgmParams.stepXY = computeDownscale(mp, fileScale * sgmParams.scale, maxW, maxH);
        }

        autoSgmScaleStep = true;
    }
    
    // single tile case, update parameters
    if(depthMapParams.autoAdjustSmallImage && mvsUtils::hasOnlyOneTile(tileParams, mp.getMaxImageWidth(), mp.getMaxImageHeight()))
    {
        // update SGM maxTCamsPerTile
        if(sgmParams.maxTCamsPerTile < depthMapParams.maxTCams)
        {
          LOG_X("Single tile computation, override SGM maximum number of T cameras per tile (before: " << sgmParams.maxTCamsPerTile << ", now: " << depthMapParams.maxTCams << ").");
          sgmParams.maxTCamsPerTile = depthMapParams.maxTCams;
        }

        // update Refine maxTCamsPerTile
        if(refineParams.maxTCamsPerTile < depthMapParams.maxTCams)
        {
          LOG_X("Single tile computation, override Refine maximum number of T cameras per tile (before: " << refineParams.maxTCamsPerTile << ", now: " << depthMapParams.maxTCams << ").");
          refineParams.maxTCamsPerTile = depthMapParams.maxTCams;
        }

        const int maxSgmBufferWidth  = divideRoundUp(mp.getMaxImageWidth() , sgmParams.scale * sgmParams.stepXY);
        const int maxSgmBufferHeight = divideRoundUp(mp.getMaxImageHeight(), sgmParams.scale * sgmParams.stepXY);

        // update SGM step XY
        if(!autoSgmScaleStep && // user define SGM scale & stepXY
           (sgmParams.stepXY == 2) && // default stepXY
           (maxSgmBufferWidth  < tileParams.bufferWidth  * 0.5) &&
           (maxSgmBufferHeight < tileParams.bufferHeight * 0.5))
        {
          LOG_X("Single tile computation, override SGM step XY (before: " << sgmParams.stepXY  << ", now: 1).");
          sgmParams.stepXY = 1;
        }
    }
    
    // compute the maximum downscale factor
    const int maxDownscale = std::max(sgmParams.scale * sgmParams.stepXY, refineParams.scale * refineParams.stepXY);

    // check padding
    if(tileParams.padding % maxDownscale != 0)
    {
      const int padding = divideRoundUp(tileParams.padding, maxDownscale) * maxDownscale;
      LOG_X("Override tiling padding parameter (before: " << tileParams.padding << ", now: " << padding << ").");
      tileParams.padding = padding;
    }
    
    // camera list
    std::vector<int> cams;
    cams.reserve(mp.ncams);

    if(rangeSize == -1)
    {
      for(int rc = 0; rc < mp.ncams; ++rc) // process all cameras
        cams.push_back(rc);
    }
    else
    {
      if(rangeStart < 0)
      {
        LOG_ERROR("invalid subrange of cameras to process.");
        return EXIT_FAILURE;
      }
      for(int rc = rangeStart; rc < std::min(rangeStart + rangeSize, mp.ncams); ++rc)
        cams.push_back(rc);
      if(cams.empty())
      {
        LOG_INFO("No camera to process.");
        return EXIT_SUCCESS;
      }
    }

    // initialize depth map estimator
    depthMap::DepthMapEstimator depthMapEstimator(mp, tileParams, depthMapParams, sgmParams, refineParams);

    // number of GPUs to use (0 means use all GPUs)
    int nbGPUs = 0;
    // estimate depth maps
    depthMap::computeOnMultiGPUs(cams, depthMapEstimator, nbGPUs);
    
    LOG_INFO("DepthMapEstimation Done!");
    return EXIT_SUCCESS;
}
