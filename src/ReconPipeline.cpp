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
#include <camera/IntrinsicBase.hpp>
#include <featureEngine/FeatureExtractor.hpp>
#include <system/Timer.hpp>
#include <SoftVisionLog.h>

ReconPipeline ReconPipeline::GetInstance()
{
    static ReconPipeline pipeline;
    return pipeline;
}

ReconPipeline::~ReconPipeline()
{
    delete m_sfmData;
    LOG_DEBUG("ReconPipeline Destruction");
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
    
    std::map<std::string, std::string> metadata_; //TODO: fill this
    auto pView = std::make_shared<sfmData::View>(viewId,
                                                 intrinsicId,
                                                  poseId,
                                                  width,
                                                  height,
                                                  (uint8_t*)bufferData,
                                                  metadata_);
    
    pView->setFrameId(frameId);
    auto&& views = m_sfmData->views;
    views[IndexT(views.size() - 1)] = pView;
    
    auto pIntrinsic = std::make_shared<camera::IntrinsicBase>(width, height);
    auto&& intrinsics = m_sfmData->intrinsics;
    intrinsics[IndexT(intrinsics.size() - 1)] = pIntrinsic;
    
    
    
}

void ReconPipeline::SetOutputDataDir(const char* directory)
{
    printf("%p Do SetOutputDataDir ...\n", this);
    m_outputFolder = directory;
}

bool ReconPipeline::FeatureExtraction()
{
    printf("%p Do FeatureExtraction ...\n", this);
    featureEngine::FeatureExtractor extractor(*m_sfmData);
    int rangeStart = 0, rangeSize = m_sfmData->views.size();
    extractor.setRange(rangeStart, rangeSize);
    
    std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::DSPSIFT);
    {
        std::vector<feature::EImageDescriberType> imageDescriberTypes = feature::EImageDescriberType_stringToEnums(describerTypesName);

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
