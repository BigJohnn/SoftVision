#pragma once

#import <Metal/Metal.h>
// #define DEPTHMAP_TEXTURE_USE_UCHAR
#define DEPTHMAP_TEXTURE_USE_HALF
#define DEPTHMAP_TEXTURE_USE_INTERPOLATION

#ifdef DEPTHMAP_TEXTURE_USE_HALF
#include <half.hpp>
#endif

#include <depthMap/gpu/host/utils.hpp>
#include <SoftVisionLogger.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <assert.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <half.h>

namespace depthMap {

#ifdef DEPTHMAP_TEXTURE_USE_UCHAR
using CudaColorBaseType = unsigned char;
using CudaRGBA = uchar4;

#else
#ifdef DEPTHMAP_TEXTURE_USE_HALF
struct CudaRGBA { half x, y, z, w; };
using CudaColorBaseType = half;
#else
using CudaColorBaseType = float;
using CudaRGBA = float4;
#endif // DEPTHMAP_TEXTURE_USE_HALF
#endif // DEPTHMAP_TEXTURE_USE_UCHAR

/*********************************************************************************
 * forward declarations
 *********************************************************************************/

template <class Type, unsigned Dim> class CudaDeviceMemoryPitched;

/*********************************************************************************
 * CudaSizeBase
 *********************************************************************************/

template <unsigned Dim> class CudaSizeBase
{
public:
  CudaSizeBase()
  {
    #pragma unroll
    for(int i = Dim; i--;)
      size[i] = 0;
  }
  inline size_t operator[](size_t i) const { return size[i]; }
  inline size_t &operator[](size_t i) { return size[i]; }
  inline CudaSizeBase operator+(const CudaSizeBase<Dim> &s) const {
    CudaSizeBase<Dim> r;

    #pragma unroll
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] + s[i];

    return r;
  }
  inline CudaSizeBase operator-(const CudaSizeBase<Dim> &s) const {
    CudaSizeBase<Dim> r;

    #pragma unroll
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] - s[i];

    return r;
  }

//#if 0
  inline size_t getSize() const {
    size_t s = 1;

    #pragma unroll
    for(int i = Dim; i--;)
      s *= size[i];

    return s;
  }
//#endif

protected:
  size_t size[Dim];
};

template <unsigned Dim>
bool operator==(const CudaSizeBase<Dim> &s1, const CudaSizeBase<Dim> &s2)
{
  for(int i = Dim; i--;)
    if(s1[i] != s2[i])
      return false;

  return true;
}

template <unsigned Dim>
bool operator!=(const CudaSizeBase<Dim> &s1, const CudaSizeBase<Dim> &s2)
{
  for(size_t i = Dim; i--;)
    if(s1[i] != s2[i])
      return true;

  return false;
}

/*********************************************************************************
 * CudaSize
 *********************************************************************************/

template <unsigned Dim>
class CudaSize: public CudaSizeBase<Dim>
{
    CudaSize() {}
};

template <>
class CudaSize<1>: public CudaSizeBase<1>
{
public:
    CudaSize() {}
    explicit CudaSize(size_t s0) { size[0] = s0; }
};

template <>
class CudaSize<2>: public CudaSizeBase<2>
{
public:
    CudaSize() {}
    CudaSize(size_t s0, size_t s1) { size[0] = s0; size[1] = s1; }

    inline size_t x() const { return size[0]; }
    inline size_t y() const { return size[1]; }
};

template <>
class CudaSize<3>: public CudaSizeBase<3>
{
public:
    CudaSize() {}
    CudaSize(size_t s0, size_t s1, size_t s2) { size[0] = s0; size[1] = s1; size[2] = s2; }

    inline size_t x() const { return size[0]; }
    inline size_t y() const { return size[1]; }
    inline size_t z() const { return size[2]; }
};

template <unsigned Dim>
CudaSize<Dim> operator/(const CudaSize<Dim> &lhs, const float &rhs) {
  if (rhs == 0)
    fprintf(stderr, "Division by zero!!\n");
  CudaSize<Dim> out = lhs;
  for(size_t i = 0; i < Dim; ++i)
    out[i] /= rhs;

  return out;
}

template <unsigned Dim>
CudaSize<Dim> operator-(const CudaSize<Dim> &lhs, const CudaSize<Dim> &rhs) {
  CudaSize<Dim> out = lhs;
  for(size_t i = Dim; i--;)
    out[i]-= rhs[i];
  return out;
}

/*********************************************************************************
 * CudaMemorySizeBase
 *********************************************************************************/

template <class Type, unsigned Dim> class CudaMemorySizeBase
{
    CudaSize<Dim> _size;
    size_t        _pitch;
protected:
    id<MTLDevice> _device;
    id<MTLBuffer> _buffer;
public:
    CudaMemorySizeBase( )
    { }

    explicit CudaMemorySizeBase( const CudaSize<Dim> &size )
        : _size( size )
        , _pitch( size[0] * sizeof(Type) )
    { }

    /* Initialize or change the contained _size value. As a
     * convenience for the many subclasses whose pitch is always
     * size[0] * sizeof(Type), true can be passed as second
     * parameter.
     */
    void setSize( const CudaSize<Dim> &size, bool computePitch )
    {
        _size = size;
        if( computePitch )
        {
            _pitch = size[0] * sizeof(Type);
        }
    }

    /* Return the Size struct.
     * It is best to use this as an opaque type.
     */
    inline const CudaSize<Dim>& getSize() const
    {
        return _size;
    }

    /* Return the byte size of dimension 0 with padding.
     * This function may return useless info until the
     * actual pitch has been initiated by the subclass.
     */
    inline size_t getPitch() const
    {
        return _pitch;
    }

    /* Return the number of bytes that are required by the data
     * contained in the subclass. The pitch may be different.
     * For many subclasses, getBytesUnpadded() == getBytesPadded()
     */
    inline size_t getBytesUnpadded() const
    {
        size_t prod = _size[0] * sizeof(Type);
        for( int i=1; i<Dim; i++ ) prod *= _size[i];
        return prod;
    }

    /* Return the number of bytes that are required to contain
     * the data of the subclass. This considers the pitch of the
     * first dimension.
     * For many subclasses, getBytesUnpadded() == getBytesPadded()
     */
    inline size_t getBytesPadded() const
    {
        size_t prod = _pitch;
        for( int i=1; i<Dim; i++ ) prod *= _size[i];
        return prod;
    }

    /* Returns the number of items that have been allocated,
     * ignoring padding.
     */
    inline size_t getUnitsTotal( ) const
    {
        size_t prod = _size[0];
        for( int i=1; i<Dim; i++ ) prod *= _size[i];
        return prod;
    }

    /* Returns the number of items of class Type that is contained
     * in the given dimension. For dimensions >= Dim, return 1.
     */
    inline size_t getUnitsInDim( int dim ) const
    {
        return ( dim < Dim ? _size[dim] : 1 );
    }

    /* For dim 0, return the pitch.
     * For all other dimensions, return the number of units in that dimension.
     */
    inline size_t getPaddedBytesInRow( ) const
    {
        return getPitch();
    }

    /* For dim 0, return the number of meaning bytes.
     * For all other dimensions, return the number of units in that dimension.
     */
    inline size_t getUnpaddedBytesInRow( ) const
    {
        return _size[0] * sizeof(Type);
    }

    /* Return the number of bytes that are required for an n-dimensional
     * slice of the subclass, always starting at dimension 0.
     *
     * Note that "dim" itself is included in the computation.
     */
    inline size_t getBytesPaddedUpToDim( int dim ) const
    {
        size_t prod = _pitch;
        for( int i=1; i<=dim; i++ ) prod *= getUnitsInDim(i);
        return prod;
    }

protected:
    /* Use to set the pitch when it has been returned by an allocation
     * function.
     */
    inline void setPitch( size_t pitch )
    {
        _pitch = pitch;
    }

    /* Allows the child class to pass the pitch to functions such as
     * CudaMallocPitched for initialization to the true value.
     */
    inline size_t& getPitchRef()
    {
        return _pitch;
    }
};

/*********************************************************************************
 * CudaHostMemoryHeap
 *********************************************************************************/

template <class Type, unsigned Dim> class CudaHostMemoryHeap : public CudaMemorySizeBase<Type,Dim>
{
    Type* buffer = nullptr;
    
public:
    CudaHostMemoryHeap( )
        : buffer( nullptr )
    { }

    explicit CudaHostMemoryHeap(const CudaSize<Dim> &size )
        : buffer( nullptr )
    {
        allocate( size );
    }

    CudaHostMemoryHeap<Type,Dim>& operator=(const CudaHostMemoryHeap<Type,Dim>& rhs)
    {
        if( buffer != nullptr )
        {
            allocate( rhs.getSize() );
        }
        else if( this->getSize() != rhs.getSize() )
        {
            deallocate();
            allocate( rhs.getSize() );
        }

        memcpy(buffer, rhs.buffer, rhs.getBytesPadded() );
        return *this;
    }

    ~CudaHostMemoryHeap()
    {
        deallocate();
    }

    void initBuffer()
    {
        memset(buffer, 0, this->getBytesPadded());
    }

    // see below with copy() functions
    void copyFrom( const CudaDeviceMemoryPitched<Type, Dim>& src, cudaStream_t stream = 0);

//    inline Type *getBuffer()
//    {
//        return buffer;
//    }
//    inline const Type *getBuffer() const
//    {
//        return buffer;
//    }
    
    inline id<MTLBuffer> getBuffer()
    {
        return _buffer;
    }
    inline const id<MTLBuffer> getBuffer() const
    {
        return _buffer;
    }
    
    inline Type& operator()(size_t x)
    {
        return buffer[x];
    }
    inline const Type& operator()(size_t x) const
    {
        return buffer[x];
    }
    inline Type& operator()(size_t x, size_t y)
    {
        return getRow(y)[x];
    }
    inline const Type& operator()(size_t x, size_t y) const
    {
        return getRow(y)[x];
    }

    inline unsigned char* getBytePtr()
    {
        return (unsigned char*)buffer;
    }
    inline const unsigned char* getBytePtr() const
    {
        return (unsigned char*)buffer;
    }

private:
    inline Type* getRow(size_t row)
    {
        unsigned char* ptr = getBytePtr();
        ptr += row * this->getPitch();
        return (Type*)ptr;
    }
    inline const Type* getRow(size_t row) const
    {
        const unsigned char* ptr = getBytePtr();
        ptr += row * this->getPitch();
        return (Type*)ptr;
    }

public:
    void allocate( const CudaSize<Dim> &size )
    {
        this->setSize( size, true );

//        buffer = malloc(this->getBytesUnpadded());
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithLength:sizeof(this->getBytesUnpadded())
                                      options:MTLResourceStorageModeShared];
        buffer = (Type*)_buffer.contents;
//        cudaError_t err = cudaMallocHost( &buffer, this->getBytesUnpadded() );
//
//        THROW_ON_CUDA_ERROR( err, "Could not allocate pinned host memory in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
    }

    void deallocate( )
    {
        //appl. ARC
        if( buffer == nullptr ) return;
////        cudaFreeHost(buffer);
//        free(buffer);
        buffer = nullptr;
    }
};

/*********************************************************************************
 * CudaDeviceMemoryPitched
 *********************************************************************************/

template <class Type, unsigned Dim> class CudaDeviceMemoryPitched : public CudaMemorySizeBase<Type,Dim>
{
    Type* buffer = nullptr;
public:
    CudaDeviceMemoryPitched( )
        : buffer( nullptr )
    { }

    explicit CudaDeviceMemoryPitched(const CudaSize<Dim> &size)
    {
        allocate( size );
    }

    explicit CudaDeviceMemoryPitched(const CudaHostMemoryHeap<Type, Dim> &rhs)
    {
        allocate( rhs.getSize() );
        copyFrom( rhs );
    }

    ~CudaDeviceMemoryPitched()
    {
        deallocate();
    }

    CudaDeviceMemoryPitched<Type,Dim>& operator=(const CudaDeviceMemoryPitched<Type,Dim> & rhs)
    {
        if( buffer == nullptr )
        {
            allocate( rhs.getSize() );
        }
        else if( this->getSize() != rhs.getSize() )
        {
            deallocate( );
            allocate( rhs.getSize() );
        }
        copyFrom( rhs );
        return *this;
    }

    // see below with copy() functions
//    void copyFrom( const CudaDeviceMemoryPitched<Type, Dim>& src, cudaStream_t stream = 0 );
//    void copyFrom( const CudaHostMemoryHeap<Type, Dim>& src, cudaStream_t stream = 0 );
    void copyFrom( const Type* src, size_t sx, size_t sy );
    

    void copyTo( Type* dst, size_t sx, size_t sy ) const;

    id<MTLBuffer> getBuffer()
    {
        return _buffer;
    }

    const id<MTLBuffer> getBuffer() const
    {
        return _buffer;
    }

    Type& operator()(size_t x)
    {
        return buffer[x];
    }

    Type& operator()(size_t x, size_t y)
    {
        Type* row = getRow( y );
        return row[x];
    }

    inline unsigned char* getBytePtr()
    {
        return (unsigned char*)buffer;
    }

    inline const unsigned char* getBytePtr() const
    {
        return (unsigned char*)buffer;
    }

    inline Type* getRow( size_t row )
    {
        unsigned char* ptr = getBytePtr();
        ptr += row * this->getPitch();
        return (Type*)ptr;
    }

    void allocate( const CudaSize<Dim>& size )
    {
        this->setSize( size, false );

        _device = MTLCreateSystemDefaultDevice();
        if(Dim == 2)
        {
            _buffer = [_device newBufferWithLength:sizeof(this->getUnpaddedBytesInRow() * this->getUnitsInDim(1))
                                          options:MTLResourceStorageModePrivate];
            buffer = (Type*)_buffer.contents;
            
//            cudaError_t err = cudaMallocPitch<Type>(&buffer,
//                                                    &this->getPitchRef(),
//                                                    this->getUnpaddedBytesInRow(),
//                                                    this->getUnitsInDim(1) );
//            if( err != cudaSuccess )
//            {
//                int devid;
//                cudaGetDevice( &devid );
//                std::stringstream ss;
//                ss << "Could not allocate pitched device memory.\n"
//                   << "Device " << devid << " alloc " << this->getBytesUnpadded() << " bytes failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err);
//                throw std::runtime_error(ss.str());
//            }
        }
        else if(Dim == 3)
        {
            _buffer = [_device newBufferWithLength:sizeof(this->getUnpaddedBytesInRow() * this->getUnitsInDim(1) * this->getUnitsInDim(2))
                                          options:MTLResourceStorageModePrivate];
            buffer = (Type*)_buffer.contents;
            
//            cudaExtent extent;
//            extent.width  = this->getUnpaddedBytesInRow();
//            extent.height = this->getUnitsInDim(1);
//            extent.depth  = this->getUnitsInDim(2);
//            cudaPitchedPtr pitchDevPtr;
//            cudaError_t err = cudaMalloc3D(&pitchDevPtr, extent);
//            if( err != cudaSuccess )
//            {
//                int devid;
//                cudaGetDevice( &devid );
//                size_t bytes = this->getBytesUnpadded();
//                size_t sx    = this->getUnitsInDim(0);
//                size_t sy    = this->getUnitsInDim(1);
//                size_t sz    = this->getUnitsInDim(2);
//                std::stringstream ss;
//                ss << "Could not allocate 3D device memory.\n"
//                   << "Device " << devid << " alloc "
//                                    << sx << "x" << sy << "x" << sz << "x" << sizeof(Type) << " = "
//                                    << bytes << " bytes ("
//                << (int)(bytes/1024.0f/1024.0f) << " MB) failed in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err);
//                throw std::runtime_error(ss.str());
//            }
//
//            buffer = (Type*)pitchDevPtr.ptr;
//            this->setPitch( pitchDevPtr.pitch );

            LOG_DEBUG("GPU 3D allocation: " << this->getUnitsInDim(0) << "x" << this->getUnitsInDim(1) << "x" << this->getUnitsInDim(2) << ", type size=" << sizeof(Type);
            LOG_DEBUG("                 : " << this->getBytesUnpadded() << ", padded=" << this->getBytesPadded() << ", wasted=" << this->getBytesPadded() - this->getBytesUnpadded() << ", wasted ratio=" << ((this->getBytesPadded() - this->getBytesUnpadded()) / double(this->getBytesUnpadded())) * 100.0 << "%");
        }
        else
        {
            throw std::runtime_error("CudaDeviceMemoryPitched does not support " + std::to_string(Dim) + " dimensions.");
        }
    }

    void deallocate()
    {
        if( buffer == nullptr ) return;
//
//        cudaError_t err = cudaFree(buffer);
//        if( err != cudaSuccess )
//        {
//            std::stringstream ss;
//            ss << "CudaDeviceMemoryPitched: Device free failed, " << cudaGetErrorString(err);
//            throw std::runtime_error(ss.str());
//        }
//
        buffer = nullptr;
    }
};

/*********************************************************************************
 * CudaDeviceMemory
 *********************************************************************************/

template <class Type> class CudaDeviceMemory : public CudaMemorySizeBase<Type,1>
{
    Type* buffer = nullptr;
public:

    CudaDeviceMemory()
      : buffer( nullptr )
    { }

    explicit CudaDeviceMemory(const size_t size)
    {
        allocate( size );
    }

    explicit inline CudaDeviceMemory(const CudaHostMemoryHeap<Type,1> &rhs)
    {
        allocate( rhs.getSize() );
        copy(*this, rhs);
    }

    // constructor with synchronous copy
    CudaDeviceMemory(const Type* inbuf, const size_t size )
    {
        allocate( size );
        copyFrom( inbuf, size );
    }

    // constructor with asynchronous copy
    CudaDeviceMemory(const Type* inbuf, const size_t size, cudaStream_t stream )
    {
        allocate( size );
        copyFrom( inbuf, size, stream );
    }

    ~CudaDeviceMemory()
    {
        deallocate( );
    }

    CudaDeviceMemory<Type> & operator=(const CudaDeviceMemory<Type> & rhs)
    {
        if( buffer == nullptr )
        {
            allocate( rhs.getSize() );
        }
        else if( this->getSize() != rhs.getSize() )
        {
            deallocate( );
            allocate( rhs.getSize() );
        }
        copy(*this, rhs);
        return *this;
    }

    Type *getBuffer()
    {
        return buffer;
    }
    const Type *getBuffer() const
    {
        return buffer;
    }

    unsigned char* getBytePtr()
    {
        return (unsigned char*)buffer;
    }
    const unsigned char* getBytePtr() const
    {
        return (unsigned char*)buffer;
    }

    void allocate( const CudaSize<1> &size )
    {
        this->setSize( size, true );

        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithLength:sizeof(this->getBytesUnpadded())
                                      options:MTLResourceStorageModePrivate];
        buffer = (Type*)_buffer.contents;
            
//        cudaError_t err = cudaMalloc(&buffer, this->getBytesUnpadded() );
//
//        THROW_ON_CUDA_ERROR( err, "Could not allocate pinned host memory in " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(err) );
    }
    void allocate( const size_t size )
    {
        allocate(CudaSize<1>(size));
    }

    void deallocate()
    {
        if(buffer == nullptr)
          return;

//        CHECK_CUDA_RETURN_ERROR(cudaFree(buffer));

        buffer = nullptr;
    }

    void copyFrom( const Type* inbuf, const size_t num )
    {
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithBytes:inbuf length:sizeof(this->getBytesUnpadded())
                                      options:MTLResourceStorageModePrivate];
        buffer = (Type*)_buffer.contents;
        
//        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
//        cudaError_t err = cudaMemcpy( buffer,
//                                      inbuf,
//                                      num * sizeof(Type),
//                                      kind );
//
//        THROW_ON_CUDA_ERROR( err, "Failed to copy from flat host buffer to CudaDeviceMemory in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) );
    }

//    void copyFrom( const Type* inbuf, const size_t num, cudaStream_t stream )
//    {
//        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
//        cudaError_t err = cudaMemcpyAsync( buffer,
//                                           inbuf,
//                                           num * sizeof(Type),
//                                           kind,
//                                           stream );
//
//        THROW_ON_CUDA_ERROR( err, "Failed to copy from flat host buffer to CudaDeviceMemory in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) );
//    }
};

/*********************************************************************************
 * copyFrom member functions
 *********************************************************************************/

//template<class Type, unsigned Dim>
//void CudaDeviceMemoryPitched<Type, Dim>::copyFrom(const CudaDeviceMemoryPitched<Type, Dim>& src, cudaStream_t stream)
//{
//    const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
//    cudaError_t err;
//    if(Dim == 1)
//    {
//        if( stream == 0 )
//            err = cudaMemcpy( this->getBytePtr(),
//                              src.getBytePtr(),
//                              src.getUnpaddedBytesInRow(),
//                              kind );
//        else
//            err = cudaMemcpyAsync( this->getBytePtr(),
//                                   src.getBytePtr(),
//                                   src.getUnpaddedBytesInRow(),
//                                   kind,
//                                   stream );
//
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//    else if(Dim >= 2)
//    {
//        size_t number_of_rows = 1;
//        for( int i=1; i<Dim; i++ ) number_of_rows *= src.getUnitsInDim(i);
//
//        if( stream == 0 )
//            err = cudaMemcpy2D( this->getBytePtr(),
//                                this->getPitch(),
//                                src.getBytePtr(),
//                                src.getPitch(),
//                                src.getUnpaddedBytesInRow(),
//                                number_of_rows,
//                                kind );
//        else
//            err = cudaMemcpy2DAsync( this->getBytePtr(),
//                                     this->getPitch(),
//                                     src.getBytePtr(),
//                                     src.getPitch(),
//                                     src.getUnpaddedBytesInRow(),
//                                     number_of_rows,
//                                     kind,
//                                     stream );
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//}
//
//template<class Type, unsigned Dim>
//void CudaDeviceMemoryPitched<Type, Dim>::copyFrom( const CudaHostMemoryHeap<Type, Dim>& src, cudaStream_t stream )
//{
//    const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
//    cudaError_t err;
//    if(Dim == 1)
//    {
//        if( stream == 0 )
//            err = cudaMemcpy( this->getBytePtr(),
//                              src.getBytePtr(),
//                              src.getBytesUnpadded(),
//                              kind );
//        else
//            err = cudaMemcpyAsync( this->getBytePtr(),
//                                   src.getBytePtr(),
//                                   src.getBytesUnpadded(),
//                                   kind,
//                                   stream );
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//    else if(Dim >= 2)
//    {
//        size_t number_of_rows = 1;
//        for( int i=1; i<Dim; i++ ) number_of_rows *= src.getUnitsInDim(i);
//
//        if( stream == 0 )
//            err = cudaMemcpy2D( this->getBytePtr(),
//                                this->getPitch(),
//                                src.getBytePtr(),
//                                src.getPitch(),
//                                src.getUnpaddedBytesInRow(),
//                                number_of_rows,
//                                kind );
//        else
//            err = cudaMemcpy2DAsync( this->getBytePtr(),
//                                     this->getPitch(),
//                                     src.getBytePtr(),
//                                     src.getPitch(),
//                                     src.getUnpaddedBytesInRow(),
//                                     number_of_rows,
//                                     kind,
//                                     stream );
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//}


template<class Type, unsigned Dim>
void CudaDeviceMemoryPitched<Type, Dim>::copyFrom( const Type* src, size_t sx, size_t sy )
{
    if(Dim == 2)
    {
        const size_t src_pitch  = sx * sizeof(Type);
        const size_t src_width  = sx * sizeof(Type);
        const size_t src_height = sy;
        
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithBytes:src length:sizeof(src_width * src_height)
                                      options:MTLResourceStorageModePrivate];
        
        this->getBytePtr() = (Type*)_buffer.contents;
        
//        cudaError_t err = cudaMemcpy2D(this->getBytePtr(),
//                                       this->getPitch(),
//                                       src,
//                                       src_pitch,
//                                       src_width,
//                                       src_height,
//                                       cudaMemcpyHostToDevice);
//
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
    }
}

//template<class Type, unsigned Dim>
//void CudaHostMemoryHeap<Type, Dim>::copyFrom(const CudaDeviceMemoryPitched<Type, Dim>& src, cudaStream_t stream)
//{
//    const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
//    cudaError_t err;
//    if(Dim == 1)
//    {
//        if( stream == 0 )
//            err = cudaMemcpy( this->getBytePtr(),
//                              src.getBytePtr(),
//                              src.getUnpaddedBytesInRow(),
//                              kind );
//        else
//            err = cudaMemcpyAsync( this->getBytePtr(),
//                                   src.getBytePtr(),
//                                   src.getUnpaddedBytesInRow(),
//                                   kind,
//                                   stream );
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//    else if(Dim >= 2)
//    {
//        size_t number_of_rows = 1;
//        for( int i=1; i<Dim; i++ ) number_of_rows *= src.getUnitsInDim(i);
//
//        if( stream == 0 )
//            err = cudaMemcpy2D( this->getBytePtr(),
//                                this->getPitch(),
//                                src.getBytePtr(),
//                                src.getPitch(),
//                                src.getUnpaddedBytesInRow(),
//                                number_of_rows,
//                                kind );
//        else
//            err = cudaMemcpy2DAsync( this->getBytePtr(),
//                                     this->getPitch(),
//                                     src.getBytePtr(),
//                                     src.getPitch(),
//                                     src.getUnpaddedBytesInRow(),
//                                     number_of_rows,
//                                     kind,
//                                     stream );
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
//    }
//}

template<class Type, unsigned Dim>
void CudaDeviceMemoryPitched<Type, Dim>::copyTo( Type* dst, size_t sx, size_t sy ) const
{
    if(Dim == 2)
    {
        const size_t dst_pitch  = sx * sizeof(Type);
        const size_t dst_width  = sx * sizeof(Type);
        const size_t dst_height = sy;
        
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithBytes:this->getBytePtr() length:sizeof(dst_width * dst_height)
                                      options:MTLResourceStorageModePrivate];
        
        dst = (Type*)_buffer.contents;
        
//        cudaError_t err = cudaMemcpy2D(dst,
//                                       dst_pitch,
//                                       this->getBytePtr(),
//                                       this->getPitch(),
//                                       dst_width,
//                                       dst_height,
//                                       cudaMemcpyDeviceToHost);
//
//        THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
    }
}

/*********************************************************************************
 * copy functions
 *********************************************************************************/

template<class Type, unsigned Dim> void copy(CudaHostMemoryHeap<Type, Dim>& dst, const CudaDeviceMemoryPitched<Type, Dim>& src)
{
    dst.copyFrom( src );
}

template<class Type> void copy(CudaHostMemoryHeap<Type,1>& _dst, const CudaDeviceMemory<Type>& _src)
{
    _device = MTLCreateSystemDefaultDevice();
    
    _buffer = [_device newBufferWithBytes:_src.getBytePtr() length:_dst.getUnpaddedBytesInRow()/*check this*/
                                  options:MTLResourceStorageModePrivate];
    
    _dst.getBytePtr() = (Type*)_buffer.contents;
                
//  cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
//  cudaError_t err = cudaMemcpy(_dst.getBytePtr(),
//                               _src.getBytePtr(),
//                               _dst.getUnpaddedBytesInRow(),
//                               kind);
//  THROW_ON_CUDA_ERROR(err, "Failed to copy from CudaHostMemoryHeap to CudaDeviceMemory");
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const CudaHostMemoryHeap<Type, Dim>& _src)
{
    _dst.copyFrom( _src );
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
    _dst.copyFrom( _src );
}

template<class Type> void copy(CudaDeviceMemory<Type>& _dst, const CudaHostMemoryHeap<Type,1>& _src)
{
    _device = MTLCreateSystemDefaultDevice();
    
    _buffer = [_device newBufferWithBytes:_src.getBytePtr() length:_src.getUnpaddedBytesInRow()
                                  options:MTLResourceStorageModePrivate];
    
    _dst.getBytePtr() = (Type*)_buffer.contents;
                
//  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
//  cudaError_t err = cudaMemcpy(_dst.getBytePtr(),
//                               _src.getBytePtr(),
//                               _src.getUnpaddedBytesInRow(),
//                               kind);
//  THROW_ON_CUDA_ERROR(err, "Failed to copy from CudaHostMemoryHeap to CudaDeviceMemory");
}

template<class Type> void copy(CudaDeviceMemory<Type>& _dst, const Type* buffer, const size_t numelems )
{
    _dst.copyFrom( buffer, numelems );
}

template<class Type, unsigned Dim> void copy(Type* _dst, size_t sx, size_t sy, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
  if(Dim == 2) {
      
      _device = MTLCreateSystemDefaultDevice();
      
      _buffer = [_device newBufferWithBytes:_src.getBytePtr() length:sx * sizeof (Type) * sy
                                    options:MTLResourceStorageModePrivate];
      
      _dst = (Type*)_buffer.contents;
      
//    cudaError_t err = cudaMemcpy2D(_dst,
//                                   sx * sizeof (Type),
//                                   _src.getBytePtr(),
//                                   _src.getPitch(),
//                                   sx * sizeof (Type),
//                                   sy,
//                                   cudaMemcpyDeviceToHost);
//    THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
  }
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& dst, const Type* src, size_t sx, size_t sy)
{
    dst.copyFrom( src, sx, sy );
}

template<class Type, unsigned Dim> void copy(Type* _dst, size_t sx, size_t sy, size_t sz, const CudaDeviceMemoryPitched<Type, Dim>& _src)
{
    if(Dim >= 3)
    {
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithBytes:_src.getBytePtr() length:sx * sizeof (Type) * sy * sz
                                      options:MTLResourceStorageModePrivate];
        
        _dst = (Type*)_buffer.contents;
        
//      cudaError_t err = cudaMemcpy2D( _dst,
//                                      sx * sizeof (Type),
//                                      _src.getBytePtr(),
//                                      _src.getPitch(),
//                                      sx * sizeof (Type),
//                                      sy * sz,
//                                      cudaMemcpyDeviceToHost);
//      THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
    }
}

template<class Type, unsigned Dim> void copy(CudaDeviceMemoryPitched<Type, Dim>& _dst, const Type* _src, size_t sx, size_t sy, size_t sz)
{
    if(Dim >= 3)
    {
      const size_t src_pitch      = sx * sizeof(Type);
      const size_t width_in_bytes = sx * sizeof(Type);
      const size_t height         = sy * sz;
        
        _device = MTLCreateSystemDefaultDevice();
        
        _buffer = [_device newBufferWithBytes:_src length:width_in_bytes * height
                                      options:MTLResourceStorageModePrivate];
        
        _dst.getBytePtr() = (Type*)_buffer.contents;
        
//      cudaError_t err = cudaMemcpy2D( _dst.getBytePtr(),
//                                      _dst.getPitch(),
//                                      _src,
//                                      src_pitch,
//                                      width_in_bytes,
//                                      height,
//                                      cudaMemcpyHostToDevice);
//      THROW_ON_CUDA_ERROR(err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ")");
    }
}

//template<class Type> void copy2D( Type* dst, size_t sx, size_t sy,
//                                  Type* src, size_t src_pitch,
//                                  cudaStream_t stream )
//{
//    const size_t dst_pitch      = sx * sizeof(Type);
//    const size_t width_in_bytes = sx * sizeof(Type);
//    const size_t height         = sy;
//
//    cudaError_t err = cudaMemcpy2DAsync( dst,
//                                         dst_pitch,
//                                         src,
//                                         src_pitch,
//                                         width_in_bytes,
//                                         height,
//                                         cudaMemcpyDeviceToHost,
//                                         stream );
//    THROW_ON_CUDA_ERROR( err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) << ")" );
//}

template<class Type> void copy2D( Type* dst, size_t sx, size_t sy,
                                  Type* src, size_t src_pitch )
{
    const size_t dst_pitch      = sx * sizeof(Type);
    const size_t width_in_bytes = sx * sizeof(Type);
    const size_t height         = sy;
                
    _device = MTLCreateSystemDefaultDevice();
    
    _buffer = [_device newBufferWithBytes:_src length:width_in_bytes * height
                                  options:MTLResourceStorageModePrivate];
    
    _dst = (Type*)_buffer.contents;
                
//    cudaError_t err = cudaMemcpy2D( dst,
//                                    dst_pitch,
//                                    src,
//                                    src_pitch,
//                                    width_in_bytes,
//                                    height,
//                                    cudaMemcpyDeviceToHost );
//    THROW_ON_CUDA_ERROR( err, "Failed to copy (" << __FILE__ << " " << __LINE__ << ", " << cudaGetErrorString(err) << ")" );
}

/*
 * @struct CudaTexture
 * @brief Support class to maintain a buffer texture in gpu memory.
 *
 * @tparam Type the buffer type
 *
 * @tparam subpixelInterpolation enable subpixel interpolation
 *   - can have a large performance impact on some graphic cards
 *   - could be critical for quality during SGM in small resolution
 *
 * @tparam normalizedCoords enable normalized coordinates
 *   - if true  addressed (x,y) in [0, 1]
 *   - if false addressed (x,y) in [width, height]
 */
template <class Type, bool subpixelInterpolation, bool normalizedCoords>
struct CudaTexture
{
    id<MTLTexture> textureObj;

//    cudaTextureObject_t textureObj = 0;

    CudaTexture(CudaDeviceMemoryPitched<Type, 2>& buffer_dmp)
    {
        MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];

        if(normalizedCoords) {
            descriptor.pixelFormat = MTLPixelFormatR16Float;
        }
        else {
            descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
        }

        // Indicate that each pixel has a blue, green, red, and alpha channel, where each channel is
        // an 8-bit unsigned normalized value (i.e. 0 maps to 0.0 and 255 maps to 1.0)
        

        // Set the pixel dimensions of the texture
//        textureDescriptor.width = image.width;
//        textureDescriptor.height = image.height;

        // Create the texture from the device by using the descriptor
        
        
        descriptor.textureType      = MTLTextureType2D;
        descriptor.width            = buffer_dmp.getSize()[0];
        descriptor.height           = buffer_dmp.getSize()[1];
        
//        descriptor.depth            = texture.depth;
//        descriptor.mipmapLevelCount = texture.mipmapLevelCount;
//        descriptor.arrayLength      = texture.arrayLength;
//        descriptor.sampleCount      = texture.sampleCount;
        descriptor.storageMode      = MTLStorageModePrivate;
        textureObj = [_device newTextureWithDescriptor:descriptor];
        
        //TODO: sampler settings MTLSamplerDescriptor
        MTLSamplerDescriptor* samplerDesc = [MTLSamplerDescriptor new];
        samplerDesc.normalizedCoordinates = normalizedCoords;
        samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLSamplerState> state = [device makeSamplerState(descriptor: samplerDesc)];
//        cudaTextureDesc  texDesc;
//        memset(&texDesc, 0, sizeof(cudaTextureDesc));
//        texDesc.normalizedCoords = normalizedCoords;
//        texDesc.addressMode[0] = cudaAddressModeClamp;
//        texDesc.addressMode[1] = cudaAddressModeClamp;
//        texDesc.addressMode[2] = cudaAddressModeClamp;
//        texDesc.readMode = cudaReadModeElementType;
//        texDesc.filterMode = (subpixelInterpolation) ? cudaFilterModeLinear : cudaFilterModePoint;
//
//        cudaResourceDesc resDesc;
//        resDesc.resType = cudaResourceTypePitch2D;
//        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<Type>();
//        resDesc.res.pitch2D.devPtr = buffer_dmp.getBuffer();
//        resDesc.res.pitch2D.width = buffer_dmp.getSize()[0];
//        resDesc.res.pitch2D.height = buffer_dmp.getSize()[1];
//        resDesc.res.pitch2D.pitchInBytes = buffer_dmp.getPitch();

        // create texture object
        // note: we only have to do this once
//        CHECK_CUDA_RETURN_ERROR(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));
    }

//    ~CudaTexture()
//    {
//        CHECK_CUDA_RETURN_ERROR_NOEXCEPT(cudaDestroyTextureObject(textureObj));
//    }
};

struct CudaRGBATexture
{
//    cudaTextureObject_t textureObj = 0;
    id<MTLTexture> textureObj;

    CudaRGBATexture(CudaDeviceMemoryPitched<CudaRGBA, 2>& buffer_dmp)
    {
        MTLTextureDescriptor * descriptor = [MTLTextureDescriptor new];

        descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
        descriptor.textureType      = MTLTextureType2D;
        descriptor.width            = buffer_dmp.getSize()[0];
        descriptor.height           = buffer_dmp.getSize()[1];
        descriptor.storageMode      = MTLStorageModePrivate;
        textureObj = [_device newTextureWithDescriptor:descriptor];
        
        //TODO: sampler settings MTLSamplerDescriptor
        MTLSamplerDescriptor* samplerDesc = [MTLSamplerDescriptor new];
        samplerDesc.normalizedCoordinates = normalizedCoords;
        samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLSamplerState> state = [device makeSamplerState(descriptor: samplerDesc)];
//        cudaTextureDesc  texDesc;
//        memset(&texDesc, 0, sizeof(cudaTextureDesc));
//        texDesc.normalizedCoords = false;
//        texDesc.addressMode[0] = cudaAddressModeClamp;
//        texDesc.addressMode[1] = cudaAddressModeClamp;
//        texDesc.addressMode[2] = cudaAddressModeClamp;
//
//#if defined(DEPTHMAP_TEXTURE_USE_UCHAR) && defined(DEPTHMAP_TEXTURE_USE_INTERPOLATION)
//        texDesc.readMode = cudaReadModeNormalizedFloat; // uchar to float [0:1], see tex2d_float4 function
//
//#else
//        texDesc.readMode = cudaReadModeElementType;
//#endif
//
//#if defined DEPTHMAP_TEXTURE_USE_INTERPOLATION
//        texDesc.filterMode = cudaFilterModeLinear;
//#else
//        texDesc.filterMode = cudaFilterModePoint;
//#endif
//
//        cudaResourceDesc resDesc;
//        resDesc.resType = cudaResourceTypePitch2D;
//
//#ifdef DEPTHMAP_TEXTURE_USE_HALF
//        resDesc.res.pitch2D.desc = cudaCreateChannelDescHalf4();
//#else
//        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<CudaRGBA>();
//#endif
//
//        resDesc.res.pitch2D.devPtr = buffer_dmp.getBuffer();
//        resDesc.res.pitch2D.width = buffer_dmp.getSize()[0];
//        resDesc.res.pitch2D.height = buffer_dmp.getSize()[1];
//        resDesc.res.pitch2D.pitchInBytes = buffer_dmp.getPitch();
//
//        // create texture object
//        // note: we only have to do this once
//        CHECK_CUDA_RETURN_ERROR(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));
    }

//    ~CudaRGBATexture()
//    {
//        CHECK_CUDA_RETURN_ERROR_NOEXCEPT(cudaDestroyTextureObject(textureObj));
//    }
};

} // namespace depthMap


