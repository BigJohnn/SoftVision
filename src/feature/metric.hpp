// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Hamming.hpp"

#include <numeric/Accumulator.hpp>
//#include <config.hpp>

//#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_SSE)
//#include <SoftVisionLog.h>
//#include <xmmintrin.h>
//#endif

#include <cstddef>


namespace feature {

/// Squared Euclidean distance functor.
template<class T>
struct L2_Simple
{
  typedef T ElementType;
  typedef typename Accumulator<T>::Type ResultType;

  template <typename Iterator1, typename Iterator2>
  inline ResultType operator()(Iterator1 a, Iterator2 b, size_t size) const
  {
    ResultType result = ResultType();
    ResultType diff;
    for(size_t i = 0; i < size; ++i ) {
      diff = *a++ - *b++;
      result += diff*diff;
    }
    return result;
  }
};

/// Squared Euclidean distance functor (vectorized version)
template<class T>
struct L2_Vectorized
{
  typedef T ElementType;
  typedef typename Accumulator<T>::Type ResultType;

  template <typename Iterator1, typename Iterator2>
  inline ResultType operator()(Iterator1 a, Iterator2 b, size_t size) const
  {
    ResultType result = ResultType();
    ResultType diff0, diff1, diff2, diff3;
    Iterator1 last = a + size;
    Iterator1 lastgroup = last - 3;

    // Process 4 items with each loop for efficiency.
    while (a < lastgroup) {
      diff0 = a[0] - b[0];
      diff1 = a[1] - b[1];
      diff2 = a[2] - b[2];
      diff3 = a[3] - b[3];
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;
    }
    // Process last 0-3 pixels.  Not needed for standard vector lengths.
    while (a < last) {
      diff0 = *a++ - *b++;
      result += diff0 * diff0;
    }
    return result;
  }
};

}  // namespace feature
