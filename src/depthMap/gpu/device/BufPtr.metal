
//#pragma once

namespace depthMap {

template <typename T>
class BufPtr
{
public:

    BufPtr(device T* ptr, int pitch)
    : _ptr( (device unsigned char*)ptr )
    , _pitch( pitch )
    {}

    inline device T* ptr()  { return (device T*)(_ptr); }
    inline device T* row(int y) { return (device T*)(_ptr + y * _pitch); }
    inline device T& at(int x, int y) { return row(y)[x]; }

    inline const device T* ptr() const { return (device const T*)(_ptr); }
    inline const device T* row(int y) const { return (device const T*)(_ptr + y * _pitch); }
    inline const thread T& at(int x, int y) const { return row(y)[x]; }

private:
    BufPtr();
    BufPtr(const device BufPtr&);
    device BufPtr& operator*=(const device BufPtr&);

    device unsigned char* const _ptr;
    const int _pitch;
};

/**
* @brief
* @param[int] ptr
* @param[int] pitch raw length of a line in bytes
* @param[int] x
* @param[int] y
* @return
*/
template <typename T>
inline device T* get2DBufferAt(device const T* ptr, int pitch, int x, int y)
{
    return BufPtr<T>(ptr,pitch).at(x,y);
}
    
//template <typename T>
//static inline device T* get3DBufferAt_h(device T* ptr, int spitch, int pitch, int x, int y, int z)
//{
//    return ((device T*)(((device char*)ptr) + z * spitch + y * pitch)) + x;
//}
//
//template <typename T>
//static inline const device T* get3DBufferAt_h(const device T* ptr, int spitch, int pitch, int x, int y, int z)
//{
//    return ((const device T*)(((const device char*)ptr) + z * spitch + y * pitch)) + x;
//}

} // namespace depthMap


