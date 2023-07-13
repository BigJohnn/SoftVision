//
//  IntrinsicBase.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef IntrinsicBase_hpp
#define IntrinsicBase_hpp

namespace camera {
/**
 * @brief Basis class for all intrinsic parameters of a camera.
 *
 * Store the image size & define all basis optical modelization of a camera
 */
class IntrinsicBase
{
public:
    explicit IntrinsicBase(unsigned int width, unsigned int height, const char* serialNumber = "") :
    _w(width), _h(height), _serialNumber(serialNumber)
    {
    }
    
    virtual ~IntrinsicBase() = default;
    
    unsigned int _w = 0;
    unsigned int _h = 0;
    double _sensorWidth = 36.0;
    double _sensorHeight = 24.0;
    const char* _serialNumber = nullptr;
};
}

#endif /* IntrinsicBase_hpp */
