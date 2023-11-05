
#pragma once

#include <simd/simd.h>

struct ROI_d {
    ROI_d(float left, float top, float right, float bottom) {
        lt = simd_make_float2(left, top);
        rb = simd_make_float2(right, bottom);
    }
    vector_float2 lt;
    vector_float2 rb;
};
