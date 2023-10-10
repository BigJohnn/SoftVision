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

    imageSpec.attribute("AliceVision:ColorSpace",
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
            imageSpec.attribute("AliceVision:storageDataType",
                                EStorageDataType_enumToString(options.getStorageDataType()));
        }

        const std::string storageDataTypeStr = imageSpec.get_string_attribute("AliceVision:storageDataType", EStorageDataType_enumToString(EStorageDataType::HalfFinite));
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

}
