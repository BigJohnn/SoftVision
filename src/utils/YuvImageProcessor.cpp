//
//  YuvImageProcessor.cpp
//  SoftVision
//
//  Created by HouPeihong on 2023/8/28.
//

#include <YuvImageProcessor.h>
#include "libyuv.h"

using namespace libyuv;

void Convert2Portrait(int in_w, int in_h, const uint8_t* in_buffer,
                     int &out_w, int &out_h, uint8_t*out_buffer)
{
    out_w = in_h;
    out_h = in_w;
    
    int sz = in_w * in_h;
    auto* i420_y = new uint8_t[sz];
    auto* i420_u = new uint8_t[sz/4];
    auto* i420_v = new uint8_t[sz/4];
    ARGBToI420(in_buffer, in_w * 4, i420_y, in_w, i420_u, in_w/2, i420_v, in_w/2, in_w, in_h);
    
    auto* i420_y_rot90 = new uint8_t[sz];
    auto* i420_u_rot90 = new uint8_t[sz/4];
    auto* i420_v_rot90 = new uint8_t[sz/4];
    I420Rotate(i420_y, in_w, i420_u, in_w/2, i420_v, in_w/2, i420_y_rot90, in_h, i420_u_rot90, in_h/2, i420_v_rot90, in_h/2, in_w, in_h, RotationMode::kRotate270);
    
    I420Mirror(i420_y_rot90, in_h, i420_u_rot90, in_h/2, i420_v_rot90, in_h/2,
               i420_y, in_h, i420_u, in_h/2, i420_v, in_h/2, in_h, in_w);
    
    I420ToARGB(i420_y, in_h, i420_u, in_h/2, i420_v, in_h/2, out_buffer, in_h * 4, in_h, in_w);
    
    delete i420_y;
    delete i420_u;
    delete i420_v;
    
    delete i420_y_rot90;
    delete i420_u_rot90;
    delete i420_v_rot90;
}
