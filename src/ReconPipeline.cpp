//
//  camerainit.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/13.
//

#include <ReconPipeline.h>
//#include <featureEngine/FeatureExtractor.hpp>

//#include <feature/feature.hpp>
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

#include <matchingImageCollection/IImageCollectionMatcher.hpp>
#include <matchingImageCollection/matchingCommon.hpp>

std::vector<std::vector<uint8_t>> ReconPipeline::m_cachedBuffers;

#ifdef SOFTVISION_DEBUG
#include <thread>
#endif

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
    
    auto pView = std::make_shared<sfmData::View>(viewId,
                                                 intrinsicId,
                                                    poseId,
                                                  width,
                                                  height,
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

    //TODO:
    
//    if(!sfm::loadRegionsPerView(regionPerView, m_sfmData, featuresFolders, describerTypes, filter))
//    {
//        LOG_ERROR("Invalid regions in '%s'", sfmDataFilename.c_str());
//        return EXIT_FAILURE;
//    }

    
    return true;
}
