//
//  io.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/31.
//

#include "io.hpp"
#include <utils/strUtils.hpp>
#include <SoftVisionLog.h>
#include <algorithm>
#include <OpenImageIO/imagebufalgo.h>
#include <half.h>
#include <utils/fileUtil.hpp>
#include <image/dcp.hpp>
#include <stl/mapUtils.hpp>

namespace image {

void byteBuffer2EigenMatrix(int w, int h, const uint8_t* imageBuf, Image<RGBAColor>& image)
{
    // TODO: impl this ...
    image.resize(w,h);
    memcpy(image.data(), imageBuf, w * h * 4 * sizeof(uint8_t));
}

std::string EImageFileType_enumToString(const EImageFileType imageFileType)
{
  switch(imageFileType)
  {
    case EImageFileType::JPEG:  return "jpg";
    case EImageFileType::PNG:   return "png";
    case EImageFileType::TIFF:  return "tif";
    case EImageFileType::EXR:   return "exr";
    case EImageFileType::NONE:  return "none";
  }
  throw std::out_of_range("Invalid EImageType enum");
}

std::string EImageExrCompression_enumToString(const EImageExrCompression exrCompression)
{
    switch (exrCompression)
    {
    case EImageExrCompression::None:  return "none";
    case EImageExrCompression::Auto:  return "auto";
    case EImageExrCompression::RLE:   return "rle";
    case EImageExrCompression::ZIP:   return "zip";
    case EImageExrCompression::ZIPS:  return "zips";
    case EImageExrCompression::PIZ:   return "piz";
    case EImageExrCompression::PXR24: return "pxr24";
    case EImageExrCompression::B44:   return "b44";
    case EImageExrCompression::B44A:  return "b44a";
    case EImageExrCompression::DWAA:  return "dwaa";
    case EImageExrCompression::DWAB:  return "dwab";
    }
    throw std::out_of_range("Invalid EImageExrCompression enum");
}

template<typename T>
void writeImage(const std::string& path,
                oiio::TypeDesc typeDesc,
                int nchannels,
                const Image<T>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList(),
                const oiio::ROI& displayRoi = oiio::ROI(),
                const oiio::ROI& pixelRoi = oiio::ROI())
{
//    const fs::path bPath = fs::path(path);
    const std::string extension = utils::GetFileExtension(path);
//    const std::string tmpPath =  (bPath.parent_path() / bPath.stem()).string() + "." + fs::unique_path().string() + extension;
    const bool isEXR = (extension == ".exr");
    //const bool isTIF = (extension == ".tif");
    const bool isJPG = (extension == ".jpg");
    const bool isPNG = (extension == ".png");

    auto toColorSpace = options.getToColorSpace();
    auto fromColorSpace = options.getFromColorSpace();

    if (toColorSpace == EImageColorSpace::AUTO)
    {
        if (isJPG || isPNG)
            toColorSpace = EImageColorSpace::SRGB;
        else
            toColorSpace = EImageColorSpace::LINEAR;
    }

    LOG_X("[IO] Write Image: " << path << "\n"
                        << "\t- width: " << image.Width() << "\n"
                        << "\t- height: " << image.Height() << "\n"
                        << "\t- channels: " << nchannels);

    oiio::ImageSpec imageSpec(image.Width(), image.Height(), nchannels, typeDesc);
    imageSpec.extra_attribs = metadata; // add custom metadata

    imageSpec.attribute("jpeg:subsampling", "4:4:4");           // if possible, always subsampling 4:4:4 for jpeg

    std::string compressionMethod = "none";
    if (isEXR)
    {
        const std::string methodName = EImageExrCompression_enumToString(options.getExrCompressionMethod());
        const int compressionLevel = options.getExrCompressionLevel();
        std::string suffix = "";
        switch (options.getExrCompressionMethod())
        {
            case EImageExrCompression::Auto:
                compressionMethod = "zips";
                break;
            case EImageExrCompression::DWAA:
            case EImageExrCompression::DWAB:
                if (compressionLevel > 0) suffix = ":" + std::to_string(compressionLevel);
                compressionMethod = methodName + suffix;
                break;
            case EImageExrCompression::ZIP:
            case EImageExrCompression::ZIPS:
                if (compressionLevel > 0) suffix = ":" + std::to_string(std::min(compressionLevel, 9));
                compressionMethod = methodName + suffix;
                break;
            default:
                compressionMethod = methodName;
                break;
        }
    }
    else if (isJPG)
    {
        LOG_ERROR("is JPG, TODO: ...");
//        if (options.getJpegCompress())
//        {
//            compressionMethod = "jpeg:" + std::to_string(std::clamp(options.getJpegQuality(), 0, 100));
//        }
    }

    imageSpec.attribute("compression", compressionMethod);

    if(displayRoi.defined() && isEXR)
    {
        imageSpec.set_roi_full(displayRoi);
    }

    if(pixelRoi.defined() && isEXR)
    {
        imageSpec.set_roi(pixelRoi);
    }

    imageSpec.attribute("SoftVision:ColorSpace",
                        (toColorSpace == EImageColorSpace::NO_CONVERSION)
                            ? EImageColorSpace_enumToString(fromColorSpace) : EImageColorSpace_enumToString(toColorSpace));
  
    const oiio::ImageBuf imgBuf = oiio::ImageBuf(imageSpec, const_cast<T*>(image.data())); // original image buffer
    const oiio::ImageBuf* outBuf = &imgBuf;  // buffer to write
        
    oiio::ImageBuf colorspaceBuf = oiio::ImageBuf(imageSpec, const_cast<T*>(image.data())); // buffer for image colorspace modification
    if ((fromColorSpace == toColorSpace) || (toColorSpace == EImageColorSpace::NO_CONVERSION))
    {
        // Do nothing. Note that calling imageAlgo::colorconvert() will copy the source buffer
        // even if no conversion is needed.
    }
    else if ((toColorSpace == EImageColorSpace::ACES2065_1) || (toColorSpace == EImageColorSpace::ACEScg) ||
             (fromColorSpace == EImageColorSpace::ACES2065_1) || (fromColorSpace == EImageColorSpace::ACEScg) ||
             (fromColorSpace == EImageColorSpace::REC709))
    {
//        const auto colorConfigPath = getAliceVisionOCIOConfig();
//        if (colorConfigPath.empty())
//        {
            throw std::runtime_error("ALICEVISION_ROOT is not defined, OCIO config file cannot be accessed.");
//        }
//        oiio::ColorConfig colorConfig(colorConfigPath);
//        oiio::ImageBufAlgo::colorconvert(colorspaceBuf, *outBuf,
//                                         EImageColorSpace_enumToOIIOString(fromColorSpace),
//                                         EImageColorSpace_enumToOIIOString(toColorSpace), true, "", "",
//                                         &colorConfig);
//        outBuf = &colorspaceBuf;
    }
    else
    {
        oiio::ImageBufAlgo::colorconvert(colorspaceBuf, *outBuf, EImageColorSpace_enumToOIIOString(fromColorSpace), EImageColorSpace_enumToOIIOString(toColorSpace));
        outBuf = &colorspaceBuf;
    }

    oiio::ImageBuf formatBuf;  // buffer for image format modification
    if(isEXR)
    {
        // Storage data type may be saved as attributes to formats that support it and then come back
        // as metadata to this function. Therefore we store the storage data type to attributes if it
        // is set and load it from attributes if it isn't set.
        if (options.getStorageDataType() != EStorageDataType::Undefined)
        {
            imageSpec.attribute("SoftVision:storageDataType",
                                EStorageDataType_enumToString(options.getStorageDataType()));
        }

        const std::string storageDataTypeStr = imageSpec.get_string_attribute("SoftVision:storageDataType", EStorageDataType_enumToString(EStorageDataType::HalfFinite));
        EStorageDataType storageDataType  = EStorageDataType_stringToEnum(storageDataTypeStr);

        if (storageDataType == EStorageDataType::Auto)
        {
            if (containsHalfFloatOverflow(*outBuf))
            {
                storageDataType = EStorageDataType::Float;
            }
            else
            {
                storageDataType = EStorageDataType::Half;
            }
            LOG_X("writeImage storageDataTypeStr: " << storageDataTypeStr);
        }

        if (storageDataType == EStorageDataType::HalfFinite)
        {
            oiio::ImageBufAlgo::clamp(colorspaceBuf, *outBuf, -HALF_MAX, HALF_MAX);
            outBuf = &colorspaceBuf;
        }

        if (storageDataType == EStorageDataType::Half ||
            storageDataType == EStorageDataType::HalfFinite)
        {
            formatBuf.copy(*outBuf, oiio::TypeDesc::HALF); // override format, use half instead of float
            outBuf = &formatBuf;
        }
    }

    auto&& outBufFlip = OpenImageIO_v2_5_2::ImageBufAlgo::flip(*outBuf);
    // write image
    if(!outBufFlip.write(path))
        throw("Can't write output image file '" + path + "'.");

    // rename temporary filename
//    fs::rename(tmpPath, path);
}


template<typename T>
void writeImageNoFloat(const std::string& path,
                oiio::TypeDesc typeDesc,
                const Image<T>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList())
{
//  const fs::path bPath = fs::path(path);
  const std::string extension = utils::GetFileExtension(path);
//  const std::string tmpPath =  (bPath.parent_path() / bPath.stem()).string() + "." + fs::unique_path().string() + extension;
  const bool isEXR = (extension == ".exr");
  //const bool isTIF = (extension == ".tif");
  const bool isJPG = (extension == ".jpg");
  const bool isPNG = (extension == ".png");

  auto imageColorSpace = options.getToColorSpace();
  if(imageColorSpace == EImageColorSpace::AUTO)
  {
    if(isJPG || isPNG)
      imageColorSpace = EImageColorSpace::SRGB;
    else
      imageColorSpace = EImageColorSpace::LINEAR;
  }

  oiio::ImageSpec imageSpec(image.Width(), image.Height(), 1, typeDesc);
  imageSpec.extra_attribs = metadata; // add custom metadata

  imageSpec.attribute("jpeg:subsampling", "4:4:4");           // if possible, always subsampling 4:4:4 for jpeg
  imageSpec.attribute("compression", isEXR ? "zips" : "none"); // if possible, set compression (zips for EXR, none for the other)

  const oiio::ImageBuf imgBuf = oiio::ImageBuf(imageSpec, const_cast<T*>(image.data())); // original image buffer
  const oiio::ImageBuf* outBuf = &imgBuf;  // buffer to write

  oiio::ImageBuf formatBuf;  // buffer for image format modification
  if(isEXR)
  {
    
    formatBuf.copy(*outBuf, typeDesc); // override format, use half instead of float
    outBuf = &formatBuf;
  }

  
  // write image
  if(!outBuf->write(path))
    throw std::runtime_error("Can't write output image file '" + path + "'.");

  // rename temporary filename
//  fs::rename(tmpPath, path);
}


void writeImage(const std::string& path, const Image<unsigned char>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImageNoFloat(path, oiio::TypeDesc::UINT8, image, options, metadata);
}

void writeImage(const std::string& path, const Image<RGBAColor>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImage(path, oiio::TypeDesc::UINT8, 4, image, options, metadata);
}

void writeImage(const std::string& path, const Image<int>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImageNoFloat(path, oiio::TypeDesc::INT32, image, options, metadata);
}

void writeImage(const std::string& path, const Image<IndexT>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImageNoFloat(path, oiio::TypeDesc::UINT32, image, options, metadata);
}

void writeImage(const std::string& path,
                const Image<float>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata,
                const oiio::ROI& displayRoi,
                const oiio::ROI& pixelRoi)
{
    writeImage(path, oiio::TypeDesc::FLOAT, 1, image, options, metadata, displayRoi, pixelRoi);
}

void writeImage(const std::string& path,
                const Image<RGBAfColor>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata,
                const oiio::ROI& displayRoi,
                const oiio::ROI& pixelRoi)
{
    writeImage(path, oiio::TypeDesc::FLOAT, 4, image, options, metadata, displayRoi, pixelRoi);
}

void writeImage(const std::string& path,
                const Image<RGBfColor>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata,
                const oiio::ROI& displayRoi,
                const oiio::ROI& pixelRoi)
{
    writeImage(path, oiio::TypeDesc::FLOAT, 3, image, options, metadata, displayRoi, pixelRoi);
}

void writeImage(const std::string& path, const Image<RGBColor>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImage(path, oiio::TypeDesc::UINT8, 3, image, options, metadata);
}

void writeImageWithFloat(const std::string& path, const Image<unsigned char>& image,
                         const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImage(path, oiio::TypeDesc::UINT8, 1, image, options, metadata);
}

void writeImageWithFloat(const std::string& path, const Image<int>& image,
                         const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImage(path, oiio::TypeDesc::INT32, 1, image, options, metadata);
}

void writeImageWithFloat(const std::string& path, const Image<IndexT>& image,
                         const ImageWriteOptions& options, const oiio::ParamValueList& metadata)
{
    writeImage(path, oiio::TypeDesc::UINT32, 1, image, options, metadata);
}

std::string EStorageDataType_enumToString(const EStorageDataType dataType)
{
    switch (dataType)
    {
    case EStorageDataType::Float:  return "float";
    case EStorageDataType::Half:   return "half";
    case EStorageDataType::HalfFinite:  return "halfFinite";
    case EStorageDataType::Auto:   return "auto";
    case EStorageDataType::Undefined: return "undefined";
    }
    throw std::out_of_range("Invalid EStorageDataType enum");
}

EStorageDataType EStorageDataType_stringToEnum(const std::string& dataType)
{
    std::string type = dataType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower); //tolower

    if (type == "float") return EStorageDataType::Float;
    if (type == "half") return EStorageDataType::Half;
    if (type == "halffinite") return EStorageDataType::HalfFinite;
    if (type == "auto") return EStorageDataType::Auto;
    if (type == "undefined") return EStorageDataType::Undefined;

    throw std::out_of_range("Invalid EStorageDataType: " + dataType);
}

bool containsHalfFloatOverflow(const oiio::ImageBuf& image)
{
    auto stats = oiio::ImageBufAlgo::computePixelStats(image);

    for(auto maxValue: stats.max)
    {
        if(maxValue > HALF_MAX)
            return true;
    }
    for (auto minValue: stats.min)
    {
        if (minValue < -HALF_MAX)
            return true;
    }
    return false;
}

bool isSupportedUndistortFormat(const std::string &ext)
{
    static const std::array<std::string, 6> supportedExtensions = {
        ".jpg", ".jpeg", ".png",  ".tif", ".tiff", ".exr"
    };
    const auto start = supportedExtensions.begin();
    const auto end = supportedExtensions.end();
    return(std::find(start, end, utils::to_lower_copy(ext)) != end);
}

oiio::ParamValueList readImageMetadata(const std::string& path, int& width, int& height)
{
    const auto spec = readImageSpec(path);
    width = spec.width;
    height = spec.height;
    return spec.extra_attribs;
}

oiio::ImageSpec readImageSpec(const std::string& path)
{
  oiio::ImageSpec configSpec;
#if OIIO_VERSION >= (10000 * 2 + 100 * 4 + 12) // OIIO_VERSION >= 2.4.12
    // To disable the application of the orientation, we need the PR https://github.com/OpenImageIO/oiio/pull/3669,
    // so we can disable the auto orientation and keep the metadata.
    configSpec.attribute("raw:user_flip", 0); // disable auto rotation of the image buffer but keep exif metadata orientation valid
#endif

  std::unique_ptr<oiio::ImageInput> in(oiio::ImageInput::open(path, &configSpec));

  if(!in)
    throw std::runtime_error("Can't find/open image file '" + path + "'.");

  oiio::ImageSpec spec = in->spec();

  in->close();

  return spec;
}

oiio::ParamValueList readImageMetadata(const std::string& path)
{
    return readImageSpec(path).extra_attribs;
}

template<typename T>
void readImage(const std::string& path,
               oiio::TypeDesc format,
               int nchannels,
               Image<T>& image,
               const ImageReadOptions& imageReadOptions)
{
    ALICEVISION_LOG_DEBUG("[IO] Read Image: " << path);

    // check requested channels number
    if (nchannels == 0)
        ALICEVISION_THROW_ERROR("Requested channels is 0. Image file: '" + path + "'.");
    if (nchannels == 2)
        ALICEVISION_THROW_ERROR("Load of 2 channels is not supported. Image file: '" + path + "'.");

    if(!utils::exists(path))
        ALICEVISION_THROW_ERROR("No such image file: '" << path << "'.");

    oiio::ImageSpec configSpec;

    const bool isRawImage = isRawFormat(path);
    image::DCPProfile::Triple neutral = {1.0,1.0,1.0};

    if (isRawImage)
    {
        if ((imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpLinearProcessing) ||
            (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpMetadata))
        {
            oiio::ParamValueList imgMetadata = readImageMetadata(path);
            std::string cam_mul = "";
            if (!imgMetadata.getattribute("raw:cam_mul", cam_mul))
            {
                cam_mul = "{1024, 1024, 1024, 1024}";
                ALICEVISION_LOG_WARNING("[readImage]: cam_mul metadata not available, the openImageIO version might be too old (>= 2.4.5.0 requested for dcp management).");
            }

            std::vector<float> v_mult;
            size_t last = 1;
            size_t next = 1;
            while ((next = cam_mul.find(",", last)) != std::string::npos)
            {
                v_mult.push_back(std::stof(cam_mul.substr(last, next - last)));
                last = next + 1;
            }
            v_mult.push_back(std::stof(cam_mul.substr(last, cam_mul.find("}", last) - last)));

            for (int i = 0; i < 3; i++)
            {
                neutral[i] = v_mult[i] / v_mult[1];
            }
        }

        ALICEVISION_LOG_TRACE("Neutral from camera = {" << neutral[0] << ", " << neutral[1] << ", " << neutral[2] << "}");

        // libRAW configuration
        // See https://openimageio.readthedocs.io/en/master/builtinplugins.html#raw-digital-camera-files
        // and https://www.libraw.org/docs/API-datastruct-eng.html#libraw_raw_unpack_params_t for raw:balance_clamped and raw:adjust_maximum_thr behavior

#if OIIO_VERSION >= (10000 * 2 + 100 * 4 + 12) // OIIO_VERSION >= 2.4.12
        // To disable the application of the orientation, we need the PR https://github.com/OpenImageIO/oiio/pull/3669,
        // so we can disable the auto orientation and keep the metadata.
        configSpec.attribute("raw:user_flip", 0); // disable auto rotation of the image buffer but keep exif metadata orientation valid
#endif

        if (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::None)
        {
            if (imageReadOptions.workingColorSpace != EImageColorSpace::NO_CONVERSION)
            {
                ALICEVISION_THROW_ERROR("Working color space must be set to \"no_conversion\" if raw color interpretation is set to \"none\"");
            }

            float user_mul[4] = {1,1,1,1};

            configSpec.attribute("raw:auto_bright", 0); // disable exposure correction
            configSpec.attribute("raw:use_camera_wb", 0); // no white balance correction
            configSpec.attribute("raw:user_mul", oiio::TypeDesc(oiio::TypeDesc::FLOAT, 4), user_mul); // no neutralization
            configSpec.attribute("raw:use_camera_matrix", 0); // do not use embeded color profile if any
            configSpec.attribute("raw:ColorSpace", "raw"); // use raw data
            configSpec.attribute("raw:HighlightMode", imageReadOptions.highlightMode);
            configSpec.attribute("raw:balance_clamped", (imageReadOptions.highlightMode == 0) ? 1 : 0);
            configSpec.attribute("raw:adjust_maximum_thr", static_cast<float>(1.0)); // Use libRaw default value: values above 75% of max are clamped to max.
            configSpec.attribute("raw:Demosaic", imageReadOptions.demosaicingAlgo);
        }
        else if (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::LibRawNoWhiteBalancing)
        {
            configSpec.attribute("raw:auto_bright", imageReadOptions.rawAutoBright); // automatic exposure correction
            configSpec.attribute("raw:Exposure", imageReadOptions.rawExposureAdjustment); // manual exposure adjustment
            configSpec.attribute("raw:use_camera_wb", 0); // no white balance correction
            configSpec.attribute("raw:use_camera_matrix", 1); // do not use embeded color profile if any, except for dng files
            configSpec.attribute("raw:ColorSpace", "Linear"); // use linear colorspace with sRGB primaries
            configSpec.attribute("raw:HighlightMode", imageReadOptions.highlightMode);
            configSpec.attribute("raw:balance_clamped", (imageReadOptions.highlightMode == 0) ? 1 : 0);
            configSpec.attribute("raw:adjust_maximum_thr", static_cast<float>(1.0)); // Use libRaw default value: values above 75% of max are clamped to max.
            configSpec.attribute("raw:Demosaic", imageReadOptions.demosaicingAlgo);
        }
        else if (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::LibRawWhiteBalancing)
        {
            configSpec.attribute("raw:auto_bright", imageReadOptions.rawAutoBright); // automatic exposure correction
            configSpec.attribute("raw:Exposure", imageReadOptions.rawExposureAdjustment); // manual exposure adjustment
            configSpec.attribute("raw:use_camera_wb", 1); // white balance correction
            configSpec.attribute("raw:use_camera_matrix", 1); // do not use embeded color profile if any, except for dng files
            configSpec.attribute("raw:ColorSpace", "Linear"); // use linear colorspace with sRGB primaries
            configSpec.attribute("raw:HighlightMode", imageReadOptions.highlightMode);
            configSpec.attribute("raw:balance_clamped", (imageReadOptions.highlightMode == 0) ? 1 : 0);
            configSpec.attribute("raw:adjust_maximum_thr", static_cast<float>(1.0)); // Use libRaw default value: values above 75% of max are clamped to max.
            configSpec.attribute("raw:Demosaic", imageReadOptions.demosaicingAlgo);
        }
        else if (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpLinearProcessing)
        {
            if (imageReadOptions.colorProfileFileName.empty())
            {
                ALICEVISION_THROW_ERROR("A DCP color profile is required but cannot be found");
            }
            float user_mul[4] = { static_cast<float>(neutral[0]),static_cast<float>(neutral[1]),static_cast<float>(neutral[2]),static_cast<float>(neutral[1]) };
            if (imageReadOptions.doWBAfterDemosaicing)
            {
                for (int i = 0; i < 4; ++i)
                {
                    user_mul[i] = 1.0;
                }
            }
            configSpec.attribute("raw:auto_bright", imageReadOptions.rawAutoBright); // automatic exposure correction
            configSpec.attribute("raw:Exposure", imageReadOptions.rawExposureAdjustment); // manual exposure adjustment
            configSpec.attribute("raw:use_camera_wb", 0); // No White balance correction => user_mul is used
            configSpec.attribute("raw:user_mul", oiio::TypeDesc(oiio::TypeDesc::FLOAT, 4), user_mul);
            configSpec.attribute("raw:use_camera_matrix", 0); // do not use embeded color profile if any
            configSpec.attribute("raw:ColorSpace", "raw");
            configSpec.attribute("raw:HighlightMode", imageReadOptions.highlightMode);
            configSpec.attribute("raw:balance_clamped", (imageReadOptions.highlightMode == 0) ? 1 : 0);
            configSpec.attribute("raw:adjust_maximum_thr", static_cast<float>(1.0)); // Use libRaw default value: values above 75% of max are clamped to max.
            configSpec.attribute("raw:Demosaic", imageReadOptions.demosaicingAlgo);
        }
        else if (imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpMetadata)
        {
            if (imageReadOptions.colorProfileFileName.empty())
            {
                ALICEVISION_THROW_ERROR("A DCP color profile is required but cannot be found");
            }
            float user_mul[4] = { static_cast<float>(neutral[0]),static_cast<float>(neutral[1]),static_cast<float>(neutral[2]),static_cast<float>(neutral[1]) };
            if (imageReadOptions.doWBAfterDemosaicing)
            {
                for (int i = 0; i < 4; ++i)
                {
                    user_mul[i] = 1.0;
                }
            }
            configSpec.attribute("raw:auto_bright", 0); // disable exposure correction
            configSpec.attribute("raw:use_camera_wb", 0); // no white balance correction
            configSpec.attribute("raw:user_mul", oiio::TypeDesc(oiio::TypeDesc::FLOAT, 4), user_mul); // no neutralization
            configSpec.attribute("raw:use_camera_matrix", 0); // do not use embeded color profile if any
            configSpec.attribute("raw:ColorSpace", "raw"); // use raw data
            configSpec.attribute("raw:HighlightMode", imageReadOptions.highlightMode);
            configSpec.attribute("raw:balance_clamped", (imageReadOptions.highlightMode == 0) ? 1 : 0);
            configSpec.attribute("raw:adjust_maximum_thr", static_cast<float>(1.0)); // Use libRaw default value: values above 75% of max are clamped to max.
            configSpec.attribute("raw:Demosaic", imageReadOptions.demosaicingAlgo);
        }
        else
        {
            ALICEVISION_THROW_ERROR("[image] readImage: invalid rawColorInterpretation " << ERawColorInterpretation_enumToString(imageReadOptions.rawColorInterpretation) << ".");
        }
    }

    oiio::ImageBuf inBuf(path, 0, 0, NULL, &configSpec);

    inBuf.read(0, 0, true, oiio::TypeDesc::FLOAT); // force image convertion to float (for grayscale and color space convertion)

    if(!inBuf.initialized())
        ALICEVISION_THROW_ERROR("Failed to open the image file: '" << path << "'.");

    // check picture channels number
    if (inBuf.spec().nchannels == 0)
        ALICEVISION_THROW_ERROR("No channel in the input image file: '" + path + "'.");
    if (inBuf.spec().nchannels == 2)
        ALICEVISION_THROW_ERROR("Load of 2 channels is not supported. Image file: '" + path + "'.");

    oiio::ParamValueList imgMetadata = readImageMetadata(path);

    if (isRawImage)
    {
        // Check orientation metadata. If image is mirrored, mirror it back and update orientation metadata
        int orientation = imgMetadata.get_int("orientation", -1);

        if (orientation == 2 || orientation == 4 || orientation == 5 || orientation == 7)
        {
            // horizontal mirroring
            oiio::ImageBuf inBufMirrored = oiio::ImageBufAlgo::flop(inBuf);
            inBuf = inBufMirrored;

            orientation += (orientation == 2 || orientation == 4) ? -1 : 1;
        }
    }

    // Apply DCP profile
    if (!imageReadOptions.colorProfileFileName.empty() &&
        imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpLinearProcessing)
    {
        image::DCPProfile dcpProfile(imageReadOptions.colorProfileFileName);

        //oiio::ParamValueList imgMetadata = readImageMetadata(path);
        std::string cam_mul = "";
        if (!imgMetadata.getattribute("raw:cam_mul", cam_mul))
        {
            cam_mul = "{1024, 1024, 1024, 1024}";
            ALICEVISION_LOG_WARNING("[readImage]: cam_mul metadata not available, the openImageIO version might be too old (>= 2.4.5.0 requested for dcp management).");
        }

        std::vector<float> v_mult;
        size_t last = 1;
        size_t next = 1;
        while ((next = cam_mul.find(",", last)) != std::string::npos)
        {
            v_mult.push_back(std::stof(cam_mul.substr(last, next - last)));
            last = next + 1;
        }
        v_mult.push_back(std::stof(cam_mul.substr(last, cam_mul.find("}", last) - last)));

        image::DCPProfile::Triple neutral;
        for (int i = 0; i < 3; i++)
        {
            neutral[i] = v_mult[i] / v_mult[1];
        }

        ALICEVISION_LOG_TRACE("Apply DCP Linear processing with neutral = " << &neutral);

        double cct = imageReadOptions.correlatedColorTemperature;

        dcpProfile.applyLinear(inBuf, neutral, cct, imageReadOptions.doWBAfterDemosaicing, imageReadOptions.useDCPColorMatrixOnly);
    }

    // color conversion
    if(imageReadOptions.workingColorSpace == EImageColorSpace::AUTO)
        ALICEVISION_THROW_ERROR("You must specify a requested color space for image file '" + path + "'.");

    // Get color space name. Default image color space is sRGB
    std::string fromColorSpaceName = (isRawImage && imageReadOptions.rawColorInterpretation == ERawColorInterpretation::DcpLinearProcessing) ? "aces2065-1" :
                                       (isRawImage ? "linear" :
                                        inBuf.spec().get_string_attribute("aliceVision:ColorSpace", inBuf.spec().get_string_attribute("oiio:ColorSpace", "sRGB")));

    ALICEVISION_LOG_TRACE("Read image " << path << " (encoded in " << fromColorSpaceName << " colorspace).");

    // Manage oiio GammaX.Y color space assuming that the gamma correction has been applied on an image with sRGB primaries.
    if (fromColorSpaceName.substr(0, 5) == "Gamma")
    {
        // Reverse gamma correction
        oiio::ImageBufAlgo::pow(inBuf, inBuf, std::stof(fromColorSpaceName.substr(5)));
        fromColorSpaceName = "linear";
    }

    DCPProfile dcpProf;
    if ((fromColorSpaceName == "no_conversion") && (imageReadOptions.workingColorSpace != EImageColorSpace::NO_CONVERSION))
    {
        ALICEVISION_LOG_INFO("Source image is in a raw color space and must be converted into " << imageReadOptions.workingColorSpace << ".");
        ALICEVISION_LOG_INFO("Check if a DCP profile is available in the metadata to be applied.");
        if (inBuf.spec().nchannels < 3)
        {
            ALICEVISION_THROW_ERROR("A DCP profile cannot be applied on an image containing less than 3 channels.");
        }

        int width, height;
        const std::map<std::string, std::string> imageMetadata = getMapFromMetadata(readImageMetadata(path, width, height));

        // load DCP metadata from metadata. An error will be thrown if all required metadata are not there.
        dcpProf.Load(imageMetadata);

        std::string cam_mul = map_has_non_empty_value(imageMetadata, "raw:cam_mul") ? imageMetadata.at("raw:cam_mul") : imageMetadata.at("SoftVision:raw:cam_mul");
        std::vector<float> v_mult;
        size_t last = 0;
        size_t next = 1;
        while ((next = cam_mul.find(",", last)) != std::string::npos)
        {
            v_mult.push_back(std::stof(cam_mul.substr(last, next - last)));
            last = next + 1;
        }
        v_mult.push_back(std::stof(cam_mul.substr(last, cam_mul.find("}", last) - last)));

        for (int i = 0; i < 3; i++)
        {
            neutral[i] = v_mult[i] / v_mult[1];
        }

        double cct = imageReadOptions.correlatedColorTemperature;

        dcpProf.applyLinear(inBuf, neutral, cct, imageReadOptions.doWBAfterDemosaicing, imageReadOptions.useDCPColorMatrixOnly);
        fromColorSpaceName = "aces2065-1";
    }

    if ((imageReadOptions.workingColorSpace == EImageColorSpace::NO_CONVERSION) ||
        (imageReadOptions.workingColorSpace == EImageColorSpace_stringToEnum(fromColorSpaceName)))
    {
        // Do nothing. Note that calling imageAlgo::colorconvert() will copy the source buffer
        // even if no conversion is needed.
    }
    else if ((imageReadOptions.workingColorSpace == EImageColorSpace::ACES2065_1) || (imageReadOptions.workingColorSpace == EImageColorSpace::ACEScg) ||
             (EImageColorSpace_stringToEnum(fromColorSpaceName) == EImageColorSpace::ACES2065_1) || (EImageColorSpace_stringToEnum(fromColorSpaceName) == EImageColorSpace::ACEScg) ||
             (EImageColorSpace_stringToEnum(fromColorSpaceName) == EImageColorSpace::REC709))
    {
//        const auto colorConfigPath = getAliceVisionOCIOConfig();
//        if (colorConfigPath.empty())
//        {
            throw std::runtime_error("ALICEVISION_ROOT is not defined, OCIO config file cannot be accessed.");
//        }
//        oiio::ImageBuf colorspaceBuf;
//        oiio::ColorConfig colorConfig(colorConfigPath);
//        oiio::ImageBufAlgo::colorconvert(colorspaceBuf, inBuf,
//            fromColorSpaceName,
//            EImageColorSpace_enumToOIIOString(imageReadOptions.workingColorSpace), true, "", "",
//            &colorConfig);
//        inBuf = colorspaceBuf;
    }
    else
    {
        oiio::ImageBuf colorspaceBuf;
        oiio::ImageBufAlgo::colorconvert(colorspaceBuf, inBuf, fromColorSpaceName, EImageColorSpace_enumToOIIOString(imageReadOptions.workingColorSpace));
        inBuf = colorspaceBuf;
    }

    // convert to grayscale if needed
    if(nchannels == 1 && inBuf.spec().nchannels >= 3)
    {
        // convertion region of interest (for inBuf.spec().nchannels > 3)
        oiio::ROI convertionROI = inBuf.roi();
        convertionROI.chbegin = 0;
        convertionROI.chend = 3;

        // compute luminance via a weighted sum of R,G,B
        // (assuming Rec709 primaries and a linear scale)
        const float weights[3] = {.2126f, .7152f, .0722f}; // To be changed if not sRGB Rec 709 Linear.
        oiio::ImageBuf grayscaleBuf;
        oiio::ImageBufAlgo::channel_sum(grayscaleBuf, inBuf, weights, convertionROI);
        inBuf.copy(grayscaleBuf);

        // TODO: if inBuf.spec().nchannels == 4: premult?
    }

    // duplicate first channel for RGB
    if (nchannels >= 3 && inBuf.spec().nchannels == 1)
    {
        oiio::ImageSpec requestedSpec(inBuf.spec().width, inBuf.spec().height, 3, format);
        oiio::ImageBuf requestedBuf(requestedSpec);
        int channelOrder[] = { 0, 0, 0 };
        float channelValues[] = { 0 /*ignore*/, 0 /*ignore*/, 0 /*ignore*/ };
        oiio::ImageBufAlgo::channels(requestedBuf, inBuf, 3, channelOrder, channelValues);
        inBuf.swap(requestedBuf);
    }

    // Add an alpha channel if needed
    if (nchannels == 4 && inBuf.spec().nchannels == 3)
    {
        oiio::ImageSpec requestedSpec(inBuf.spec().width, inBuf.spec().height, 3, format);
        oiio::ImageBuf requestedBuf(requestedSpec);
        int channelOrder[] = { 0, 1, 2, -1 /*constant value*/ };
        float channelValues[] = { 0 /*ignore*/, 0 /*ignore*/, 0 /*ignore*/, 1.0 };
        oiio::ImageBufAlgo::channels(requestedBuf, inBuf,
                                        4, // create an image with 4 channels
                                        channelOrder,
                                        channelValues); // only the 4th value is used
        inBuf.swap(requestedBuf);
    }

    // copy pixels from oiio to eigen
    image.resize(inBuf.spec().width, inBuf.spec().height, false);
    {
        oiio::ROI exportROI = inBuf.roi();
        exportROI.chbegin = 0;
        exportROI.chend = nchannels;

        inBuf.get_pixels(exportROI, format, image.data());
    }
}

template<typename T>
void readImageNoFloat(const std::string& path,
               oiio::TypeDesc format,
               Image<T>& image)
{
  oiio::ImageSpec configSpec;

  oiio::ImageBuf inBuf(path, 0, 0, NULL, &configSpec);

  inBuf.read(0, 0, true, format);

  if(!inBuf.initialized())
  {
    throw std::runtime_error("Cannot find/open image file '" + path + "'.");
  }

  // check picture channels number
  if(inBuf.spec().nchannels != 1)
  {
    throw std::runtime_error("Can't load channels of image file '" + path + "'.");
  }
    
  // copy pixels from oiio to eigen
  image.resize(inBuf.spec().width, inBuf.spec().height, false);
  {
    oiio::ROI exportROI = inBuf.roi();
    exportROI.chbegin = 0;
    exportROI.chend = 1;

    inBuf.get_pixels(exportROI, format, image.data());
  }
}


void readImage(const std::string& path, Image<float>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::FLOAT, 1, image, imageReadOptions);
}

void readImage(const std::string& path, Image<unsigned char>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::UINT8, 1, image, imageReadOptions);
}

void readImageDirect(const std::string& path, Image<unsigned char>& image)
{
  readImageNoFloat(path, oiio::TypeDesc::UINT8, image);
}

void readImageDirect(const std::string& path, Image<IndexT>& image)
{
  readImageNoFloat(path, oiio::TypeDesc::UINT32, image);
}

void readImage(const std::string& path, Image<RGBAfColor>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::FLOAT, 4, image, imageReadOptions);
}

void readImage(const std::string& path, Image<RGBAColor>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::UINT8, 4, image, imageReadOptions);
}

void readImage(const std::string& path, Image<RGBfColor>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::FLOAT, 3, image, imageReadOptions);
}

void readImage(const std::string& path, Image<RGBColor>& image, const ImageReadOptions & imageReadOptions)
{
  readImage(path, oiio::TypeDesc::UINT8, 3, image, imageReadOptions);
}

// Warning: type conversion problems from string to param value, we may lose some metadata with string maps
oiio::ParamValueList getMetadataFromMap(const std::map<std::string, std::string>& metadataMap)
{
  oiio::ParamValueList metadata;
  for(const auto& metadataPair : metadataMap)
    metadata.push_back(oiio::ParamValue(metadataPair.first, metadataPair.second));
  return metadata;
}

std::map<std::string, std::string> getMapFromMetadata(const oiio::ParamValueList& metadata)
{
    std::map<std::string, std::string> metadataMap;

    for (const auto& param : metadata)
        metadataMap.emplace(param.name().string(), param.get_string());

    return metadataMap;
}

bool isRawFormat(const std::string& path)
{
    std::unique_ptr<oiio::ImageInput> in(oiio::ImageInput::open(path));
    if(!in)
        return false;
    std::string imgFormat = in->format_name();

    return (imgFormat.compare("raw") == 0);
}


}
