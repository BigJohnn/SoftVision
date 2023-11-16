
#pragma once

namespace depthMap {

template <typename T>
class BufPtr
{
public:

    BufPtr(device T* ptr, size_t pitch)
    : _ptr( (device unsigned char*)ptr )
    , _pitch( pitch )
    {}

    inline device T* ptr()  { return (device T*)(_ptr); }
    inline device T* row(size_t y) { return (device T*)(_ptr + y * _pitch); }
    inline device T& at(size_t x, size_t y) { return row(y)[x]; }

    inline const device T* ptr() const { return (device const T*)(_ptr); }
    inline const device T* row(size_t y) const { return (device const T*)(_ptr + y * _pitch); }
    inline const device T& at(size_t x, size_t y) const { return row(y)[x]; }

private:
    BufPtr();
    BufPtr(const device BufPtr&);
    device BufPtr& operator*=(const device BufPtr&);

    device unsigned char* const _ptr;
    const size_t _pitch;
};

template <typename T>
static inline device T* get3DBufferAt_h(device T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((device T*)(((device char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
static inline const device T* get3DBufferAt_h(const device T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((const device T*)(((const device char*)ptr) + z * spitch + y * pitch)) + x;
}

} // namespace depthMap


