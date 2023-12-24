
#pragma once

#include <simd/simd.h>

//struct Range_d
//{
//    simd_uint2 val;
//};
struct ROI_d {
    vector_float2 lt;
    vector_float2 rb;
};
