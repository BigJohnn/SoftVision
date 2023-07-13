//
//  io.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/31.
//

#include "io.hpp"

namespace image {
void byteBuffer2EigenMatrix(int w, int h, const uint8_t* imageBuf, Image<RGBAColor>& image)
{
    // TODO: impl this ...
    image.resize(w,h);
    memcpy(image.data(), imageBuf, w * h * 4 * sizeof(uint8_t));
}

}
