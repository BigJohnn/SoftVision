//
//  View.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef View_hpp
#define View_hpp

#include <common/types.h>

#include <map>
#include <string>
//#include <cstdio>

namespace sfmData {

/**
 * @brief EXIF Orientation to names
 * https://jdhao.github.io/2019/07/31/image_rotation_exif_info/
 */
enum class EEXIFOrientation
{
  NONE = 1
  , REVERSED = 2
  , UPSIDEDOWN = 3
  , UPSIDEDOWN_REVERSED = 4
  , LEFT_REVERSED = 5
  , LEFT = 6
  , RIGHT_REVERSED = 7
  , RIGHT = 8
  , UNKNOWN = -1
};

struct GPSExifTags
{
    static std::string latitude();
    static std::string latitudeRef();
    static std::string longitude();
    static std::string longitudeRef();
    static std::string altitude();
    static std::string altitudeRef();
    static std::vector<std::string> all();
};

class ExposureSetting {
public:
    ExposureSetting() {}

    ExposureSetting(double shutter, double fnumber, double iso)
        : _shutter(shutter)
        , _fnumber(fnumber)
        , _iso(iso)
    {
    }

    double _shutter{-1.0};
    double _fnumber{-1.0};
    double _iso{-1.0};

    bool hasShutter() const { return _shutter > 0.0 && std::isnormal(_shutter); }
    bool hasFNumber() const { return _fnumber > 0.0 && std::isnormal(_fnumber); }
    bool hasISO() const { return _iso > 0.0 && std::isnormal(_iso); }

    bool isFullyDefined() const {
        return hasShutter() && hasFNumber() && hasISO();
    }

    bool isPartiallyDefined() const {
        return hasShutter() || hasFNumber();
    }

    double getExposure(const double referenceISO = 100.0, const double referenceFNumber = 1.0) const
    {
        const bool validShutter = hasShutter();
        const bool validFNumber = hasFNumber();

        if(!validShutter && !validFNumber)
            return -1.0;

        const bool validRefFNumber = referenceFNumber > 0.0 && std::isnormal(referenceFNumber);

        double shutter = _shutter;
        if(!validShutter)
        {
            shutter = 1.0 / 200.0;
        }
        double fnumber = _fnumber;
        // Usually we should get a valid shutter speed, but we could have invalid fnumber.
        // For instance, if there is a connection problem between the lens and the camera, all lens related option like fnumber could be invalid.
        // In this particular case, the exposure should rely only on the shutter speed.
        if(!validFNumber)
        {
            if(validRefFNumber)
                fnumber = referenceFNumber;
            else
                fnumber = 2.0;
        }
        double lReferenceFNumber = referenceFNumber;
        if(!validRefFNumber)
        {
            lReferenceFNumber = fnumber;
        }

        const double iso = _iso;
        /*
        iso = qLt / aperture^2
        isoratio = iso2 / iso1 = (qLt / aperture2^2) / (qLt / aperture1^2)
        isoratio = aperture1^2 / aperture2^2
        aperture2^2 = aperture1^2 / isoratio
        aperture2^2 = (aperture1^2 / (iso2 / iso1))
        aperture2^2 = (iso1 / iso2)
        aperture2 = sqrt(iso1 / iso2)
        */
        double iso_2_aperture = 1.0;
        if(iso > 1e-6 && referenceISO > 1e-6)
        {
            // Need to have both iso and reference iso to use it
            iso_2_aperture = std::sqrt(iso / referenceISO);
        }

        /*
        aperture = f / diameter
        aperture2 / aperture1 = diameter1 / diameter2
        (aperture2 / aperture1)^2 = (area1 / pi) / (area2 / pi)
        area2 = (aperture1 / aperture2)^2
        */
        double new_fnumber = fnumber * iso_2_aperture;
        double exp_increase = (new_fnumber / lReferenceFNumber) * (new_fnumber / lReferenceFNumber);

        // If the aperture was more important for this image, this means that it received less light than with a default aperture
        // This means also that if we want to simulate that all the image have the same aperture, we have to increase virtually th
        // light received as if the aperture was smaller. So we increase the exposure time

        // If the iso is larger than the default value, this means that it recevied more light than with a default iso
        // This means also that if we want to simulate that all the image have the same iso, we have to decrease virtually th
        // light received as if the iso was smaller. So we decrease the exposure time or equivalent, increase the aperture value

        // Checks
        // iso 20, f/2 = 2500
        // iso 40, f/2.8 = 2500

        return shutter * exp_increase;
    }
    bool operator<(const ExposureSetting& other) const { return getExposure() < other.getExposure(); }
    bool operator==(const ExposureSetting& other) const { return getExposure() == other.getExposure(); }
};


/**
 * @brief A view define an image by a string and unique indexes for
 * the view, the camera intrinsic, the pose and the subpose if the camera is part of a rig
 */
class View
{
public:
    /**
       * @brief View Constructor
       * @param[in] imagePath The image path on disk
       * @param[in] viewId The view id (use unique index)
       * @param[in] intrinsicId The intrinsic id
       * @param[in] poseId The pose id (or the rig pose id)
       * @param[in] width The image width
       * @param[in] height The image height
       * @param[in] rigId The rig id (or undefined)
       * @param[in] subPoseId The sub-pose id (or undefined)
       * @param[in] buffer The image buffer data
       * @param[in] metadata The image metadata
       */
      View(
           IndexT viewId = UndefinedIndexT,
           IndexT intrinsicId = UndefinedIndexT,
           IndexT poseId = UndefinedIndexT,
           IndexT width = 0,
           IndexT height = 0,
           std::vector<uint8_t> buffer = {},
           const std::map<std::string, std::string>& metadata = std::map<std::string, std::string>())
        : _width(width)
        , _height(height)
        , _viewId(viewId)
        , _intrinsicId(intrinsicId)
        , _poseId(poseId)
        , _buffer(buffer)
        , _metadata(metadata)
      {}
    
//    ~View(){
//        printf("View destroyed!!\n");
//    }
    
    /**
     * @brief  Set the given view image width
     * @param[in] width The given view image width
     */
    void setWidth(std::size_t width)
    {
      _width = width;
    }

    /**
     * @brief  Set the given view image height
     * @param[in] height The given view image height
     */
    void setHeight(std::size_t height)
    {
      _height = height;
    }

    /**
     * @brief Set the given view id
     * @param[in] viewId The given view id
     */
    void setViewId(IndexT viewId)
    {
      _viewId = viewId;
    }

    /**
     * @brief Set the given intrinsic id
     * @param[in] intrinsicId The given intrinsic id
     */
    void setIntrinsicId(IndexT intrinsicId)
    {
      _intrinsicId = intrinsicId;
    }

    /**
     * @brief Set the given pose id
     * @param[in] poseId The given pose id
     */
    void setPoseId(IndexT poseId)
    {
      _poseId = poseId;
    }

    /**
     * @brief setIndependantPose
     * @param independant
     */
    void setIndependantPose(bool independent)
    {
        _isPoseIndependent = independent;
    }

    
      /**
       * @brief Set the given frame id
       * @param[in] frame The given frame id
       */
      void setFrameId(IndexT frameId)
      {
        _frameId = frameId;
      }
    
    /**
       * @brief Get view image width
       * @return image width
       */
      std::size_t getWidth() const
      {
        return _width;
      }

      /**
       * @brief Get view image height
       * @return image height
       */
      std::size_t getHeight() const
      {
        return _height;
      }
    
      const uint8_t* getBuffer() const
      {
          return _buffer.data();
      }

      /**
       * @brief Get view image height
       * @return image height
       */
      std::pair<std::size_t, std::size_t> getImgSize() const
      {
        return {_width, _height};
      }

      /**
       * @brief Get the view id
       * @return view id
       */
      IndexT getViewId() const
      {
        return _viewId;
      }

      /**
       * @brief Get the intrinsic id
       * @return intrinsic id
       */
      IndexT getIntrinsicId() const
      {
        return _intrinsicId;
      }


      /**
       * @brief Get the pose id
       * @return pose id
       */
      IndexT getPoseId() const
      {
        return _poseId;
      }
    
      /**
       * @brief Add view metadata
       * @param[in] key The metadata key
       * @param[in] value The metadata value
       */
      void addMetadata(const std::string& key, const std::string& value)
      {
          _metadata[key] = value;
      }
    
    /**
       * @brief Get the metadata value as a double
       * @param[in] names List of possible names for the metadata
       * @return the metadata value as a double or -1.0 if it does not exist
       */
      double getDoubleMetadata(const std::vector<std::string>& names) const;

      /**
       * @brief Get the metadata value as a double
       * @param[in] names List of possible names for the metadata
       * @param[in] val Data to be set with the metadata value
       * @return true if the metadata is found or false if it does not exist
       */
      bool getDoubleMetadata(const std::vector<std::string>& names, double& val) const;
    
    /**
       * @brief Get the metadata value as a string
       * @param[in] names List of possible names for the metadata
       * @return the metadata value as a string or an empty string if it does not exist
       */
      const std::string& getMetadata(const std::vector<std::string>& names) const;
    
    /**
       * @brief Read a floating point value from a string. It support an integer, a floating point value or a fraction.
       * @param[in] str string with the number to evaluate
       * @return the extracted floating point value or -1.0 if it fails to convert the string
       */
      double readRealNumber(const std::string& str) const;
    
      /**
       * @brief Get the Camera Exposure Setting value.
       * For the same scene, this value is linearly proportional to the amount of light captured by the camera according to
       * the shooting parameters (shutter speed, f-number, iso).
       */
      ExposureSetting getCameraExposureSetting() const
      {
          return ExposureSetting(
              getMetadataShutter(),
              getMetadataFNumber(),
              getMetadataISO());
      }
    
      /**
       * @brief If the view is part of a camera rig, the camera can be a sub-pose of the rig pose but can also be temporarily solved independently.
       * @return true if the view is not part of a rig.
       *         true if the view is part of a rig and the camera is solved separately.
       *         false if the view is part of a rig and the camera is solved as a sub-pose of the rig pose.
       */
      bool isPoseIndependant() const
      {
        return _isPoseIndependent;
      }
    
      /**
       * @brief Get view image path
       * @return image path
       */
      const std::string& getImagePath() const
      {
        return _imagePath;
      }
    
      /**
       * @brief Get the view metadata structure
       * @return the view metadata
       */
      const std::map<std::string, std::string>& getMetadata() const
      {
        return _metadata;
      }
    
      /**
       * @brief Get the corresponding "ExposureTime" (shutter) metadata value
       * @return the metadata value float or -1 if no corresponding value
       */
      double getMetadataShutter() const
      {
          return getDoubleMetadata({"ExposureTime", "Shutter Speed Value"});
      }

      /**
       * @brief Get the corresponding "FNumber" (relative aperture) metadata value
       * @return the metadata value float or -1 if no corresponding value
       */
      double getMetadataFNumber() const
      {
          if(hasDigitMetadata({"FNumber"}))
          {
              return getDoubleMetadata({"FNumber"});
          }
          if (hasDigitMetadata({"ApertureValue", "Aperture Value"}))
          {
              const double aperture = getDoubleMetadata({"ApertureValue", "Aperture Value"});
              // fnumber = 2^(aperture/2)
              return std::pow(2.0, aperture / 2.0);
          }
          return -1;
      }

      /**
         * @brief Get the corresponding "PhotographicSensitivity" (ISO) metadata value
         * @return the metadata value int or -1 if no corresponding value
         */
      double getMetadataISO() const
      {
        return getDoubleMetadata({"Exif:PhotographicSensitivity", "PhotographicSensitivity", "Photographic Sensitivity", "ISO"});
      }
    
    /**
       * @brief Get the list of viewID referencing the source views called "Ancestors"
       * If an image is generated from multiple input images, "Ancestors" allows to keep track of the viewIDs of the original inputs views.
       * For instance, the generated view can come from the fusion of multiple LDR images into one HDR image, the fusion from multi-focus
       * stacking to get a fully focused image, fusion of images with multiple lighting to get a more diffuse lighting, etc.
       * @return list of viewID of the ancestors
       * @param[in] viewId the view ancestor id
       */
      void addAncestor(IndexT viewId)
      {
        _ancestors.push_back(viewId);
      }

      /**
      * @Brief get all ancestors for this view
      * @return ancestors
      */
      const std::vector<IndexT> & getAncestors() const
      {
        return _ancestors;
      }

      /**
       * @brief Set the given resection id
       * @param[in] resectionId The given resection id
       */
      void setResectionId(IndexT resectionId)
      {
        _resectionId = resectionId;
      }
    
      /**
       * @brief Get the frame id
       * @return frame id
       */
      IndexT getFrameId() const
      {
        return _frameId;
      }
    
      /**
       * @brief Get the resection id
       * @return resection id
       */
      IndexT getResectionId() const
      {
        return _resectionId;
      }
    
      /**
       * @brief Get the metadata value as an integer
       * @param[in] names List of possible names for the metadata
       * @return the metadata value as an integer or -1 if it does not exist
       */
      int getIntMetadata(const std::vector<std::string>& names) const;
    
      /**
       * @brief Get the corresponding "Orientation" metadata value
       * @return the enum EEXIFOrientation
       */
      EEXIFOrientation getMetadataOrientation() const
      {
        const int orientation = getIntMetadata({"Exif:Orientation", "Orientation"});
        if(orientation < 0)
          return EEXIFOrientation::UNKNOWN;
        return static_cast<EEXIFOrientation>(orientation);
      }
      /**
       * @brief Get an iterator on the map of metadata from a given name.
       */
      std::map<std::string, std::string>::const_iterator findMetadataIterator(const std::string& name) const;
    
      /**
       * @brief Return true if the metadata for longitude and latitude exist.
       * It checks that all the tags from GPSExifTags exists
       * @return true if GPS data is available
       */
      bool hasGpsMetadata() const;
    
      /**
       * @brief Return true if the given metadata name exists and is a digit
       * @param[in] names List of possible names for the metadata
       * @param[in] isPositive true if the metadata must be positive
       * @return true if the corresponding metadata value exists
       */
      bool hasDigitMetadata(const std::vector<std::string>& names, bool isPositive = true) const;
    
      /**
       * @brief Return true if the given metadata name exists
       * @param[in] names List of possible names for the metadata
       * @return true if the corresponding metadata value exists
       */
      bool hasMetadata(const std::vector<std::string>& names) const;
    
    
      /**
       * @brief Get the gps position in the absolute cartesian reference system.
       * @return The position x, y, z as a three dimensional vector.
       */
      Vec3 getGpsPositionFromMetadata() const;
    
      /**
       * @brief Get the gps position in the WGS84 reference system.
       * @param[out] lat the latitude
       * @param[out] lon the longitude
       * @param[out] alt the altitude
       */
      void getGpsPositionWGS84FromMetadata(double& lat, double& lon, double& alt) const;

      /**
       * @brief Get the gps position in the WGS84 reference system as a vector.
       * @return A three dimensional vector with latitude, logitude and altitude.
       */
      Vec3 getGpsPositionWGS84FromMetadata() const;
    
    /// image width
      std::size_t _width;
      /// image height
      std::size_t _height;
      /// view id
      IndexT _viewId;
      /// intrinsics id
      IndexT _intrinsicId;
      /// either the pose of the rig or the pose of the camera if there's no rig
      IndexT _poseId;
      /// corresponding frame id for synchronized views
      IndexT _frameId = UndefinedIndexT;
      /// map for metadata
      std::map<std::string, std::string> _metadata;
      /// list of ancestors
      std::vector<IndexT> _ancestors;
    
      IndexT _resectionId = UndefinedIndexT;
    
      std::vector<uint8_t> _buffer;
    
    /// pose independent of other view(s)
      bool _isPoseIndependent = true;
    
private:
    /// image path on disk
      std::string _imagePath;
};

}
#endif /* View_hpp */
