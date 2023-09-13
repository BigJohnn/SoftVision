//
//  FeatureExtractor.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/25.
//

#ifndef FeatureExtractor_hpp
#define FeatureExtractor_hpp

#include <sfmData/View.hpp>
#include <feature/feature.hpp>
#include <sfmData/SfMData.hpp>
#include <system/hardwareContext.hpp>
#include <image/colorspace.hpp>
#include <feature/RegionsPerView.hpp>

namespace featureEngine {
class FeatureExtractorViewJob
{
public:
    FeatureExtractorViewJob(const sfmData::View& view,
                            const std::string& outputFolder);

    ~FeatureExtractorViewJob();

//    bool useGPU() const
//    {
//        return !_gpuImageDescriberIndexes.empty();
//    }

//    bool useCPU() const
//    {
//        return !_cpuImageDescriberIndexes.empty();
//    }

    std::string getFeaturesPath(feature::EImageDescriberType imageDescriberType) const
    {
        return _outputBasename + "." + EImageDescriberType_enumToString(imageDescriberType) + ".feat";
    }

    std::string getDescriptorPath(feature::EImageDescriberType imageDescriberType) const
    {
        return _outputBasename + "." + EImageDescriberType_enumToString(imageDescriberType) + ".desc";
    }

    void setImageDescribers(
            const std::vector<std::shared_ptr<feature::ImageDescriber>>& imageDescribers);

    const sfmData::View& view() const
    {
        return _view;
    }

    std::size_t memoryConsuption() const
    {
        return _memoryConsuption;
    }

    const std::vector<std::size_t>& imageDescriberIndexes(bool useGPU) const
    {
//        return useGPU ? _gpuImageDescriberIndexes
//                      : _cpuImageDescriberIndexes;
        return _cpuImageDescriberIndexes;
    }

private:
    const sfmData::View& _view;
    std::size_t _memoryConsuption = 0;
    std::string _outputBasename;
    std::vector<std::size_t> _cpuImageDescriberIndexes;
//    std::vector<std::size_t> _gpuImageDescriberIndexes;
};

class FeatureExtractor
{
public:

    explicit FeatureExtractor(const sfmData::SfMData& sfmData);
    ~FeatureExtractor();

    void setRange(int rangeStart, int rangeSize)
    {
      _rangeStart = rangeStart;
      _rangeSize = rangeSize;
    }

    void setMasksFolder(const std::string& folder, const std::string& ext, bool invert)
    {
      _masksFolder = folder;
      _maskExtension = ext;
      _maskInvert = invert;
    }

    void setOutputFolder(const std::string& folder)
    {
      _outputFolder = folder;
    }

    void addImageDescriber(std::shared_ptr<feature::ImageDescriber>& imageDescriber)
    {
      _imageDescribers.push_back(imageDescriber);
    }

    void process(const HardwareContext & hcontext, const image::EImageColorSpace workingColorSpace = image::EImageColorSpace::SRGB);

    inline feature::RegionsPerView& getRegionsPerView() {
        return _mmapRegions;
    }
    
private:

    void computeViewJob(const FeatureExtractorViewJob& job, bool useGPU, const image::EImageColorSpace workingColorSpace = image::EImageColorSpace::SRGB);

    const sfmData::SfMData& _sfmData;
    std::vector<std::shared_ptr<feature::ImageDescriber>> _imageDescribers;
    std::string _masksFolder;
    std::string _maskExtension;
    bool _maskInvert;
    std::string _outputFolder;
    int _rangeStart = -1;
    int _rangeSize = -1;
    
    feature::RegionsPerView _mmapRegions;
};

}
#endif /* FeatureExtractor_hpp */
