//
//  YuvImageProcessor.h
//  SoftVision
//
//  Created by HouPeihong on 2023/8/28.
//

#ifndef YuvImageProcessor_h
#define YuvImageProcessor_h
#include <types.h>

/*
 * landscape bgra => portrait rgba, note: the endian is different between libyuv && ios.
 */
void Convert2Portrait(int in_w, int in_h, const uint8_t* in_buffer,
                     int &out_w, int &out_h, uint8_t*out_buffer);

#endif /* YuvImageProcessor_h */
