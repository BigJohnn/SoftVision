//
//  io.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/31.
//

#ifndef io_hpp
#define io_hpp

#include <common/types.h>
#include <image/Image.hpp>
#include <image/colorspace.hpp>

#include <OpenImageIO/paramlist.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/color.h>

namespace oiio = OIIO;

namespace image {

/**
* @brief Data type use to write the image
*/
enum class EStorageDataType
{
    Float, //< Use full floating point precision to store
    Half, //< Use half (values our of range could become inf or nan)
    HalfFinite, //< Use half, but ensures out-of-range pixels are clamps to keep finite pixel values
    Auto, //< Use half if all pixels can be stored in half without clamp, else use full float
    Undefined //< Storage data type is not defined and should be inferred from other sources
};

/**
* @brief Compression method used to write an exr image
*/
enum class EImageExrCompression
{
    None,
    Auto,
    RLE,
    ZIP,
    ZIPS,
    PIZ,
    PXR24,
    B44,
    B44A,
    DWAA,
    DWAB
};

/**
 * @brief aggregate for multiple image writing options
 */
class ImageWriteOptions
{
public:
    ImageWriteOptions() = default;

    EImageColorSpace getFromColorSpace() const { return _fromColorSpace; }
    EImageColorSpace getToColorSpace() const { return _toColorSpace; }
    EStorageDataType getStorageDataType() const { return _storageDataType; }
    EImageExrCompression getExrCompressionMethod() const { return _exrCompressionMethod; }
    int getExrCompressionLevel() const { return _exrCompressionLevel; }
    bool getJpegCompress() const { return _jpegCompress; }
    int getJpegQuality() const { return _jpegQuality; }

    ImageWriteOptions& fromColorSpace(EImageColorSpace colorSpace)
    {
        _fromColorSpace = colorSpace;
        return *this;
    }

    ImageWriteOptions& toColorSpace(EImageColorSpace colorSpace)
    {
        _toColorSpace = colorSpace;
        return *this;
    }

    ImageWriteOptions& storageDataType(EStorageDataType storageDataType)
    {
        _storageDataType = storageDataType;
        return *this;
    }

    ImageWriteOptions& exrCompressionMethod(EImageExrCompression compressionMethod)
    {
        _exrCompressionMethod = compressionMethod;
        return *this;
    }

    ImageWriteOptions& exrCompressionLevel(int compressionLevel)
    {
        _exrCompressionLevel = compressionLevel;
        return *this;
    }

    ImageWriteOptions& jpegCompress(bool compress)
    {
        _jpegCompress = compress;
        return *this;
    }

    ImageWriteOptions& jpegQuality(int quality)
    {
        _jpegQuality = quality;
        return *this;
    }

private:
    EImageColorSpace _fromColorSpace{EImageColorSpace::LINEAR};
    EImageColorSpace _toColorSpace{EImageColorSpace::AUTO};
    EStorageDataType _storageDataType{EStorageDataType::Undefined};
    EImageExrCompression _exrCompressionMethod{EImageExrCompression::Auto};
    int _exrCompressionLevel{0};
    bool _jpegCompress{true};
    int _jpegQuality{90};
};

std::string EStorageDataType_enumToString(const EStorageDataType dataType);
EStorageDataType EStorageDataType_stringToEnum(const std::string& dataType);
/**
 * @brief Available image file type for pipeline output
 */
enum class EImageFileType
{
  JPEG,
  PNG,
  TIFF,
  EXR,
  NONE
};

/**
 * @brief convert image data buffer to Image, some Eigen Matrix type.
 * @param[in] imageBuf The given buffer of the image
 * @param[out] image The output image type.
 */
void byteBuffer2EigenMatrix(int w, int h, const uint8_t* imageBuf, Image<RGBAColor>& image);

std::string EImageExrCompression_enumToString(const EImageExrCompression exrCompression);

/**
 * @brief It converts a EImageFileType enum to a string.
 * @param[in] imageFileType the EImageFileType enum to convert.
 * @return the string associated to the EImageFileType enum.
 */
std::string EImageFileType_enumToString(const EImageFileType imageFileType);

/**
 * @brief write an image with a given path and buffer
 * @param[in] path The given path to the image
 * @param[in] image The output image buffer
 */
void writeImage(const std::string& path, const Image<RGBAColor>& image,
                const ImageWriteOptions& options, const oiio::ParamValueList& metadata);

bool containsHalfFloatOverflow(const oiio::ImageBuf& image);

bool isSupportedUndistortFormat(const std::string &ext);

}
#endif /* io_hpp */
