// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace depthMap {

template <typename T>
class BufPtr
{
public:

    BufPtr(T* ptr, size_t pitch)
        : _ptr( (unsigned char*)ptr )
        , _pitch( pitch )
    {}

    inline T* ptr()  { return (T*)(_ptr); }
    inline T* row(size_t y) { return (T*)(_ptr + y * _pitch); }
    inline T& at(size_t x, size_t y) { return row(y)[x]; }

    inline const T* ptr() const { return (const T*)(_ptr); }
    inline const T* row(size_t y) const { return (const T*)(_ptr + y * _pitch); }
    inline const T& at(size_t x, size_t y) const { return row(y)[x]; }

private:
    BufPtr();
    BufPtr(const BufPtr&);
    BufPtr& operator*=(const BufPtr&);

    unsigned char* const _ptr;
    const size_t _pitch;
};


template <typename T>
static inline T* get3DBufferAt_h(T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((T*)(((char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
static inline const T* get3DBufferAt_h(const T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((const T*)(((const char*)ptr) + z * spitch + y * pitch)) + x;
}

} // namespace depthMap


