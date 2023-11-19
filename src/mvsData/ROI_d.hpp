
#pragma once

#include <simd/simd.h>

struct Range_d
{
    Range_d (int s, int t){
        begin = s;
        end = t;
    }
    unsigned int begin = 0;
    unsigned int end = 0;
};
struct ROI_d {
//    ROI_d(float left, float top, float right, float bottom) {
//        lt = simd_make_float2(left, top);
//        rb = simd_make_float2(right, bottom);
//    }
    vector_float2 lt;
    vector_float2 rb;
};
