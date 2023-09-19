//
//  camerainit.h
//  SoftVision
//
//  Created by HouPeihong on 2023/7/13.
//

#ifndef ReconPipeline_h
#define ReconPipeline_h

#include <cstdint>
#include <string>
//#include <system/hardwareContext.hpp>

namespace sfmData{
class SfMData;
}
namespace featureEngine{
class FeatureExtractor;
}

class HardwareContext;
class ReconPipeline {
    
public:
    
    ~ReconPipeline();
    
    static ReconPipeline GetInstance();
    void CameraInit(void);
    
//    "viewId":uuid,
//    "poseId":uuid,
//    "frameId":String(PhotoCaptureProcessor.frameId),
//    "data": photoData?.description as Any,
//    "width":String(describing: exif_table!["PixelXDimension"]!),
//    "height":String(describing: exif_table!["PixelYDimension"]!),
//    "metadata":exif_table as Any
    
    void AppendSfMData(uint32_t viewId,
                       uint32_t poseId,
                       uint32_t intrinsicId,
                       uint32_t frameId,
                       uint32_t width,
                       uint32_t height,
                       const uint8_t* bufferData);
    
    bool FeatureExtraction();
    
    //TODO: image matching
//    bool ImageMatching();
    
    bool FeatureMatching();
    
    bool IncrementalSFM();
    
    void SetOutputDataDir(const char* directory);
    
private:
    ReconPipeline() = default;
    
    
private:
//    static ReconPipeline* pPipeline;
    sfmData::SfMData* m_sfmData = nullptr;
    std::string m_outputFolder;
    static std::vector<std::vector<uint8_t>> m_cachedBuffers;
    std::string m_describerTypesName;
    featureEngine::FeatureExtractor* m_extractor = nullptr;
    HardwareContext* mp_hwc = nullptr;
};

#endif /* ReconPipeline_h */
