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
 * @brief Available raw processing methods
 */
enum class ERawColorInterpretation
{
    /// Debayering without any color processing
    None,
    /// Simple neutralization
    LibRawNoWhiteBalancing,
    /// Use internal white balancing from libraw
    LibRawWhiteBalancing,
    /// If DCP file is not available throw an exception
    DcpLinearProcessing,
    /// If DCP file is not available throw an exception
    DcpMetadata,
    /// Access aliceVision:rawColorInterpretation metadata to set the method
    Auto
};

std::string ERawColorInterpretation_informations();
ERawColorInterpretation ERawColorInterpretation_stringToEnum(const std::string& dataType);
std::string ERawColorInterpretation_enumToString(const ERawColorInterpretation dataType);
std::ostream& operator<<(std::ostream& os, ERawColorInterpretation dataType);
std::istream& operator>>(std::istream& in, ERawColorInterpretation& dataType);


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
 * @brief aggregate for multiple image reading options
 */
struct ImageReadOptions
{
    ImageReadOptions(EImageColorSpace colorSpace = EImageColorSpace::AUTO,
        ERawColorInterpretation rawColorInterpretation = ERawColorInterpretation::LibRawWhiteBalancing,
        const std::string& colorProfile = "", const bool useDCPColorMatrixOnly = true, const oiio::ROI& roi = oiio::ROI()) :
        workingColorSpace(colorSpace), rawColorInterpretation(rawColorInterpretation), colorProfileFileName(colorProfile), useDCPColorMatrixOnly(useDCPColorMatrixOnly),
        doWBAfterDemosaicing(false), demosaicingAlgo("AHD"), highlightMode(0), rawAutoBright(false), rawExposureAdjustment(1.0),
        correlatedColorTemperature(-1.0), subROI(roi)
    {
    }

    EImageColorSpace workingColorSpace;
    ERawColorInterpretation rawColorInterpretation;
    std::string colorProfileFileName;
    bool useDCPColorMatrixOnly;
    bool doWBAfterDemosaicing;
    std::string demosaicingAlgo;
    int highlightMode;
    bool rawAutoBright;
    float rawExposureAdjustment;
    double correlatedColorTemperature;
    //ROI for this image.
    //If the image contains an roi, this is the roi INSIDE the roi.
    oiio::ROI subROI;
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
 * @brief It returns the EImageFileType enum from a string.
 * @param[in] imageFileType the input string.
 * @return the associated EImageFileType enum.
 */
EImageFileType EImageFileType_stringToEnum(const std::string& imageFileType);

/**
 * @brief write an image with a given path and buffer
 * @param[in] path The given path to the image
 * @param[in] image The output image buffer
 */
void writeImage(const std::string& path, const Image<unsigned char>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImage(const std::string& path, const Image<int>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImage(const std::string& path, const Image<IndexT>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImage(const std::string& path,
                const Image<float>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList(),
                const oiio::ROI& displayRoi = oiio::ROI(),
                const oiio::ROI& pixelRoi = oiio::ROI());

void writeImage(const std::string& path,
                const Image<RGBAfColor>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList(),
                const oiio::ROI& displayRoi = oiio::ROI(),
                const oiio::ROI& pixelRoi = oiio::ROI());

void writeImage(const std::string& path, const Image<RGBAColor>& image, const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImage(const std::string& path,
                const Image<RGBfColor>& image,
                const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList(),
                const oiio::ROI& displayRoi = oiio::ROI(),
                const oiio::ROI& pixelRoi = oiio::ROI());

void writeImage(const std::string& path, const Image<RGBColor>& image, const ImageWriteOptions& options,
                const oiio::ParamValueList& metadata = oiio::ParamValueList());

/**
 * @brief write an image with a given path and buffer, converting to float as necessary to perform
 * intermediate calculations.
 * @param[in] path The given path to the image
 * @param[in] image The output image buffer
 */
void writeImageWithFloat(const std::string& path, const Image<unsigned char>& image,
                         const ImageWriteOptions& options,
                         const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImageWithFloat(const std::string& path, const Image<int>& image,
                         const ImageWriteOptions& options,
                         const oiio::ParamValueList& metadata = oiio::ParamValueList());

void writeImageWithFloat(const std::string& path, const Image<IndexT>& image,
                         const ImageWriteOptions& options,
                         const oiio::ParamValueList& metadata = oiio::ParamValueList());

bool containsHalfFloatOverflow(const oiio::ImageBuf& image);

bool isSupportedUndistortFormat(const std::string &ext);

oiio::ParamValueList readImageMetadata(const std::string& path, int& width, int& height);

oiio::ImageSpec readImageSpec(const std::string& path);

oiio::ParamValueList readImageMetadata(const std::string& path);

/**
 * @brief read an image with a given path and buffer
 * @param[in] path The given path to the image
 * @param[out] image The output image buffer
 * @param[in] image color space
 */
void readImage(const std::string& path, Image<float>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<unsigned char>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<IndexT>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<RGBAfColor>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<RGBAColor>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<RGBfColor>& image, const ImageReadOptions & imageReadOptions);
void readImage(const std::string& path, Image<RGBColor>& image, const ImageReadOptions & imageReadOptions);
/**
 * @brief read an image with a given path and buffer without any processing such as color conversion
 * @param[in] path The given path to the image
 * @param[out] image The output image buffer
 */
void readImageDirect(const std::string& path, Image<IndexT>& image);
void readImageDirect(const std::string& path, Image<unsigned char>& image);

/**
 * @brief convert a metadata string map into an oiio::ParamValueList
 * @param[in] metadataMap string map
 * @return oiio::ParamValueList
 */
oiio::ParamValueList getMetadataFromMap(const std::map<std::string, std::string>& metadataMap);

/**
 * @brief convert an oiio::ParamValueList into metadata string map
 * @param[in] metadata An instance of oiio::ParamValueList
 * @return std::map Metadata string map
 */
// Warning: type conversion problems from string to param value, we may lose some metadata with string maps
std::map<std::string, std::string> getMapFromMetadata(const oiio::ParamValueList& metadata);

bool isRawFormat(const std::string& path);
}
#endif /* io_hpp */
