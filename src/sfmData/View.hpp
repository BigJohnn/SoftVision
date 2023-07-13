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
           uint8_t* buffer = nullptr,
           const std::map<std::string, std::string>& metadata = std::map<std::string, std::string>())
        : _width(width)
        , _height(height)
        , _viewId(viewId)
        , _intrinsicId(intrinsicId)
        , _poseId(poseId)
        , _buffer(buffer)
        , _metadata(metadata)
      {}
    
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
    
      uint8_t* getBuffer() const
      {
          return _buffer;
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
       * @brief Get an iterator on the map of metadata from a given name.
       */
      std::map<std::string, std::string>::const_iterator findMetadataIterator(const std::string& name) const;
    
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
    
      uint8_t* _buffer;
};

}
#endif /* View_hpp */
