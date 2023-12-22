
//#pragma once
#include "../planeSweeping/similarity.hpp"

namespace depthMap {

//template <typename T>
//class BufPtr
//{
//public:
//
//    BufPtr(device T* ptr, int pitch)
//    : _ptr( (device unsigned char*)ptr )
//    , _pitch( pitch )
//    {}
//
//    inline device T* ptr()  { return (device T*)(_ptr); }
//    inline device T* row(int y) { return (device T*)(_ptr + y * _pitch); }
//    inline device T& at(int x, int y) { return row(y)[x]; }
//
//    inline const device T* ptr() const { return (device const T*)(_ptr); }
//    inline const device T* row(int y) const { return (device const T*)(_ptr + y * _pitch); }
//    inline const device T& at(int x, int y) const { return row(y)[x]; }
//
//private:
//    BufPtr();
//    BufPtr(const device BufPtr&);
//    device BufPtr& operator*=(const device BufPtr&);
//
//    device unsigned char* const _ptr;
//    const int _pitch;
//};
//
//template class BufPtr<TSimAcc>;
//template class BufPtr<TSim>;
    
/**
* @brief
* @param[int] ptr
* @param[int] pitch raw length of a line in bytes
* @param[int] x
* @param[int] y
* @return
*/
template <typename T>
device T* get2DBufferAt(device T* ptr, int pitch, int x, int y)
{
    return ((device T*)(((device unsigned char*)ptr) + y * pitch)) + x;
}
    
template device TSimAcc* get2DBufferAt(device TSimAcc* ptr, int pitch, int x, int y);
template device TSim* get2DBufferAt(device TSim* ptr, int pitch, int x, int y);
template device float* get2DBufferAt(device float* ptr, int pitch, int x, int y);

} // namespace depthMap


