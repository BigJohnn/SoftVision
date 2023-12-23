
#pragma once

#include <simd/simd.h>

struct Range_d
{
    unsigned int begin = 0;
    unsigned int end = 0;
};
struct ROI_d {
    vector_float2 lt;
    vector_float2 rb;
};
