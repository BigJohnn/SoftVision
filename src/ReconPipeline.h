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
namespace sfmData{
class SfMData;
}

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
    
    void SetOutputDataDir(const char* directory);
    
private:
    ReconPipeline() = default;
    
    
private:
//    static ReconPipeline* pPipeline;
    sfmData::SfMData* m_sfmData = nullptr;
    std::string m_outputFolder;
};

#endif /* ReconPipeline_h */
