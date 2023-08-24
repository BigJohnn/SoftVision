//
//  FeatureExtractor.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/25.
//

#include "FeatureExtractor.hpp"
#include <system/MemoryInfo.hpp>
#include <fstream>
#include <image/colorspace.hpp>
//#include <image/imageAlgo.hpp>
#include <image/io.hpp>
#include <SoftVisionLog.h>
#include <image/convertion.hpp>


#include "png.h"

namespace featureEngine {

FeatureExtractorViewJob::FeatureExtractorViewJob(const sfmData::View& view,
                                                 const std::string& outputFolder) :
    _view(view),
    _outputBasename((outputFolder + std::to_string(view.getViewId())).c_str())
//    _outputBasename(fs::path(fs::path(outputFolder) / fs::path(std::to_string(view.getViewId()))).string())
{}

FeatureExtractorViewJob::~FeatureExtractorViewJob() = default;

void FeatureExtractorViewJob::setImageDescribers(
        const std::vector<std::shared_ptr<feature::ImageDescriber>>& imageDescribers)
{
    for (std::size_t i = 0; i < imageDescribers.size(); ++i)
    {
        const std::shared_ptr<feature::ImageDescriber>& imageDescriber = imageDescribers.at(i);
        feature::EImageDescriberType imageDescriberType = imageDescriber->getDescriberType();

        std::fstream f1,f2;
        f1.open(getFeaturesPath(imageDescriberType), std::ios::in);
        f2.open(getDescriptorPath(imageDescriberType), std::ios::in);
        if (f1 && f2)
        {
            continue;
        }

        _memoryConsuption += imageDescriber->getMemoryConsumption(_view.getWidth(),
                                                                  _view.getHeight());

//        if(imageDescriber->useCuda())
//            _gpuImageDescriberIndexes.push_back(i);
//        else
            _cpuImageDescriberIndexes.push_back(i);
    }
}


FeatureExtractor::FeatureExtractor(const sfmData::SfMData& sfmData) :
    _sfmData(sfmData)
{}

FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::process(const HardwareContext & hContext, const image::EImageColorSpace workingColorSpace)
{
    size_t maxAvailableMemory = hContext.getUserMaxMemoryAvailable();
    unsigned int maxAvailableCores = hContext.getMaxThreads();
    
    // iteration on each view in the range in order
    // to prepare viewJob stack
    sfmData::Views::const_iterator itViewBegin = _sfmData.getViews().begin();
    sfmData::Views::const_iterator itViewEnd = _sfmData.getViews().end();

    if(_rangeStart != -1)
    {
        std::advance(itViewBegin, _rangeStart);
        itViewEnd = itViewBegin;
        std::advance(itViewEnd, _rangeSize);
    }

    std::size_t jobMaxMemoryConsuption = 0;

    std::vector<FeatureExtractorViewJob> cpuJobs;
//    std::vector<FeatureExtractorViewJob> gpuJobs;

    for (auto it = itViewBegin; it != itViewEnd; ++it)
    {
        const sfmData::View& view = *(it->second.get());
        FeatureExtractorViewJob viewJob(view, _outputFolder);

        viewJob.setImageDescribers(_imageDescribers);
        jobMaxMemoryConsuption = std::max(jobMaxMemoryConsuption, viewJob.memoryConsuption());

//        if (viewJob.useCPU())
            cpuJobs.push_back(viewJob);

//        if (viewJob.useGPU())
//            gpuJobs.push_back(viewJob);
    }

    if (!cpuJobs.empty())
    {
        system2::MemoryInfo memoryInformation = system2::getMemoryInfo();
        
        //Put an upper bound with user specified memory
        size_t maxMemory = std::min(memoryInformation.availableRam, maxAvailableMemory);
        size_t maxTotalMemory = std::min(memoryInformation.totalRam, maxAvailableMemory);

        LOG_INFO("Job max memory consumption for one image: %lu MB", jobMaxMemoryConsuption / (1024*1024));
//        LOG_INFO("Job max memory consumption for one image: "
//                             << jobMaxMemoryConsuption / (1024*1024) << " MB");
//        LOG_INFO("Memory information: " << std::endl << memoryInformation);
//        LOG_INFO("Memory information: ",memoryInformation)

        if (jobMaxMemoryConsuption == 0)
            throw std::runtime_error("Cannot compute feature extraction job max memory consumption.");

        // How many buffers can fit in 90% of the available RAM?
        // This is used to estimate how many jobs can be computed in parallel without SWAP.
        const std::size_t memoryImageCapacity =
                std::size_t((0.9 * maxMemory) / jobMaxMemoryConsuption);

        std::size_t nbThreads = std::max(std::size_t(1), memoryImageCapacity);
        LOG_INFO("Max number of threads regarding memory usage: %d" , nbThreads);
        const double oneGB = 1024.0 * 1024.0 * 1024.0;
        if (jobMaxMemoryConsuption > maxMemory)
        {
            LOG_INFO("The amount of RAM available is critical to extract features.");
            if (jobMaxMemoryConsuption <= maxTotalMemory)
            {
                LOG_INFO("But the total amount of RAM is enough to extract features, "
                                        "so you should close other running applications.");
//                LOG_INFO(" => " << std::size_t(std::round((double(maxTotalMemory - maxMemory) / oneGB)))
//                                        << " GB are used by other applications for a total RAM capacity of "
//                                        << std::size_t(std::round(double(maxTotalMemory) / oneGB))
//                                        << " GB.");
            }
        }
        else
        {
            if (maxMemory < 0.5 * maxTotalMemory)
            {
                LOG_INFO("More than half of the RAM is used by other applications. It would be more efficient to close them.");
//                LOG_INFO(" => "
//                                        << std::size_t(std::round(double(maxTotalMemory - maxMemory) / oneGB))
//                                        << " GB are used by other applications for a total RAM capacity of "
//                                        << std::size_t(std::round(double(maxTotalMemory) / oneGB))
//                                        << " GB.");
            }
        }

        if(maxMemory == 0)
        {
          LOG_INFO("Cannot find available system memory, this can be due to OS limitation.\n"
                                  "Use only one thread for CPU feature extraction.");
          nbThreads = 1;
        }

        // nbThreads should not be higher than the available cores
        nbThreads = std::min(static_cast<std::size_t>(maxAvailableCores), nbThreads);

        // nbThreads should not be higher than the number of jobs
        nbThreads = std::min(cpuJobs.size(), nbThreads);

        //TODO: add openmp
        LOG_INFO("# threads for extraction: %d" , nbThreads);
//        omp_set_nested(1);

#pragma omp parallel for num_threads(nbThreads)
        for (int i = 0; i < cpuJobs.size(); ++i)
            computeViewJob(cpuJobs.at(i), false, workingColorSpace);
    }

//    if (!gpuJobs.empty())
//    {
//        for (const auto& job : gpuJobs)
//            computeViewJob(job, true, workingColorSpace);
//    }
}

void FeatureExtractor::computeViewJob(const FeatureExtractorViewJob& job, bool useGPU, const image::EImageColorSpace workingColorSpace)
{
    image::Image<float> imageGrayFloat;
    
    image::Image<image::RGBAColor> imageBGRA;
    
    image::Image<unsigned char> imageGrayUChar;
    image::Image<unsigned char> mask;

    auto&& view = job.view();
    image::byteBuffer2EigenMatrix(view.getWidth(), view.getHeight(), view.getBuffer(), imageBGRA);
    
    //TODO: bytebuffer 2 png file
    {
        auto&& folder_name = _outputFolder.substr(7,_outputFolder.size()-7);
        std::string file_name = folder_name + "test.png";
        FILE *fp = fopen(file_name.c_str(), "wb");
        if (!fp)
        {
            LOG_ERROR("PNG file cannot open for write!!");
        }
        
        png_structp png_ptr = png_create_write_struct_2
        (PNG_LIBPNG_VER_STRING, NULL,
         NULL, NULL, (png_voidp)view.getBuffer(), NULL, NULL);
        
//        png_structp png_ptr = png_create_write_struct
//        (PNG_LIBPNG_VER_STRING, NULL,
//         NULL, NULL);
        
        if (!png_ptr)
            LOG_ERROR("PNG ptr null!");
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
        {
            png_destroy_write_struct(&png_ptr,
            (png_infopp)NULL);
                LOG_ERROR("info ptr null!");
        }
        
        if (setjmp(png_jmpbuf(png_ptr)))
        {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            LOG_ERROR("PNG error!");
        }
        
        
        png_init_io(png_ptr, fp);
        
//        void write_row_callback(png_ptr, png_uint_32 row,
//        int pass);
//        {
//        /* put your code here */
//        }
//        png_set_write_status_fn(png_ptr, write_row_callback);
        
        /* turn on or off filtering, and/or choose
        specific filters. You can use either a single
        PNG_FILTER_VALUE_NAME or the bitwise OR of one
        or more PNG_FILTER_NAME masks. */
//        png_set_filter(png_ptr, 0,
//        PNG_FILTER_NONE | PNG_FILTER_VALUE_NONE |
//        PNG_FILTER_SUB | PNG_FILTER_VALUE_SUB |
//        PNG_FILTER_UP | PNG_FILTER_VALUE_UP |
//        PNG_FILTER_AVG | PNG_FILTER_VALUE_AVG |
//        PNG_FILTER_PAETH | PNG_FILTER_VALUE_PAETH|
//        PNG_ALL_FILTERS);
        
        png_set_IHDR(png_ptr, info_ptr, view.getWidth(), view.getHeight(),
        8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        
//        png_set_rows(png_ptr, info_ptr, (png_bytepp)view.getBuffer());
        
        png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR, NULL);
    }
    
    
    //TODO: BGRA to RGBA!!!
    image::ConvertPixelType(imageBGRA, &imageGrayUChar);
    image::ConvertPixelType(imageGrayUChar, &imageGrayFloat);
    
//    image::readImage(job.view().getImagePath(), imageGrayFloat, workingColorSpace); // TODO: byte array to eigen matrix

    double pixelRatio = 1.0;
    job.view().getDoubleMetadata({"PixelAspectRatio"}, pixelRatio);

    if (pixelRatio != 1.0)
    {
        // Resample input image in order to work with square pixels
        const int w = imageGrayFloat.Width();
        const int h = imageGrayFloat.Height();

        const int nw = static_cast<int>(static_cast<double>(w) * pixelRatio);
        const int nh = h;

        image::Image<float> resizedInput;
//        imageAlgo::resizeImage(nw, nh, imageGrayFloat, resizedInput);
        imageGrayFloat.swap(resizedInput);
    }

//    if (!_masksFolder.empty() && fs::exists(_masksFolder))
//    {
//        const auto masksFolder = fs::path(_masksFolder);
//        const auto idMaskPath = masksFolder /
//                fs::path(std::to_string(job.view().getViewId())).replace_extension(_maskExtension);
//        const auto nameMaskPath = masksFolder /
//                fs::path(job.view().getImagePath()).filename().replace_extension(_maskExtension);
//
//        if (fs::exists(idMaskPath))
//        {
//            image::readImage(idMaskPath.string(), mask, image::EImageColorSpace::LINEAR);
//        }
//        else if (fs::exists(nameMaskPath))
//        {
//            image::readImage(nameMaskPath.string(), mask, image::EImageColorSpace::LINEAR);
//        }
//    }

    for (const auto & imageDescriberIndex : job.imageDescriberIndexes(useGPU))
    {
        const auto& imageDescriber = _imageDescribers.at(imageDescriberIndex);
        const feature::EImageDescriberType imageDescriberType = imageDescriber->getDescriberType();
        const std::string imageDescriberTypeName =
                feature::EImageDescriberType_enumToString(imageDescriberType);

        // Compute features and descriptors and export them to files
//        LOG_INFO("Extracting " << imageDescriberTypeName  << " features from view '"
//                             << job.view().getImagePath() << "' " << (useGPU ? "[gpu]" : "[cpu]"));
        LOG_INFO("Extracting %s features from view '%u' [cpu]", imageDescriberTypeName.c_str(), job.view().getViewId());
                 
        std::unique_ptr<feature::Regions> regions;
        if (imageDescriber->useFloatImage())
        {
            // image buffer use float image, use the read buffer
            imageDescriber->describe(imageGrayFloat, regions);
        }
        else
        {
            // image buffer can't use float image
            if (imageGrayUChar.Width() == 0) // the first time, convert the float buffer to uchar
                imageGrayUChar = (imageGrayFloat.GetMat() * 255.f).cast<unsigned char>();
            imageDescriber->describe(imageGrayUChar, regions);
        }

        if (pixelRatio != 1.0)
        {
            // Re-position point features on input image
            for (auto & feat : regions->Features())
            {
                feat.x() /= pixelRatio;
            }
        }

        if (mask.Height() > 0)
        {
            std::vector<feature::FeatureInImage> selectedIndices;
            for (size_t i=0, n=regions->RegionCount(); i != n; ++i)
            {
                const Vec2 position = regions->GetRegionPosition(i);
                const int x = int(position.x());
                const int y = int(position.y());

                bool masked = false;
                if (x < mask.Width() && y < mask.Height())
                {
                    if ((mask(y, x) == 0 && !_maskInvert) || (mask(y, x) != 0 && _maskInvert))
                    {
                        masked = true;
                    }
                }

                if (!masked)
                {
                    selectedIndices.push_back({IndexT(i), 0});
                }
            }

            std::vector<IndexT> out_associated3dPoint;
            std::map<IndexT, IndexT> out_mapFullToLocal;
            regions = regions->createFilteredRegions(selectedIndices, out_associated3dPoint,
                                                     out_mapFullToLocal);
        }

        imageDescriber->Save(regions.get(), job.getFeaturesPath(imageDescriberType),
                             job.getDescriptorPath(imageDescriberType));
        LOG_INFO("feature path: %s | imageDescriberType: %s", job.getFeaturesPath(imageDescriberType).c_str(),job.getDescriptorPath(imageDescriberType).c_str());
//        LOG_INFO(std::left << std::setw(6) << " " << regions->RegionCount() << " "
//                             << imageDescriberTypeName  << " features extracted from view '"
//                             << job.view().getImagePath() << "'");
        
        LOG_INFO("%d %s features extracted from view '%u'", regions->RegionCount(), imageDescriberTypeName.c_str(), job.view().getViewId());
    }
}

}
