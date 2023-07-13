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



namespace image {
/**
 * @brief convert image data buffer to Image, some Eigen Matrix type.
 * @param[in] imageBuf The given buffer of the image
 * @param[out] image The output image type.
 */
void byteBuffer2EigenMatrix(int w, int h, const uint8_t* imageBuf, Image<RGBAColor>& image);
}
#endif /* io_hpp */
