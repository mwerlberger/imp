#ifndef IMP_PIXEL_HPP
#define IMP_PIXEL_HPP

#include <cstdint>

#ifdef IMP_WITH_CUDA
#  include<cuda_runtime_api.h>
#  define CUDA_HOST __host__
#  define CUDA_DEVICE  __device__
#else
#  define CUDA_HOST
#  define CUDA_DEVICE
#endif

namespace imp {

//------------------------------------------------------------------------------
template<typename _T>
union Pixel1
{
  typedef _T T;

  struct
  {
    T x;
  };
  struct
  {
    T r;
  };
  T c[1];

  CUDA_HOST CUDA_DEVICE Pixel1() : x(0) { }
  CUDA_HOST CUDA_DEVICE Pixel1(T _x) : x(_x) { }
  CUDA_HOST CUDA_DEVICE ~Pixel1() = default;

  CUDA_HOST CUDA_DEVICE constexpr std::uint8_t numDims() {return 1;}

  CUDA_HOST CUDA_DEVICE operator T() const { return c[0]; }
  CUDA_HOST CUDA_DEVICE T& operator[](size_t i) { return c[i]; }
  CUDA_HOST CUDA_DEVICE const T& operator[](size_t i) const { return c[i]; }
  CUDA_HOST CUDA_DEVICE Pixel1<T>& operator*(const Pixel1<T>& rhs)
  {
    c[0] *= rhs[0];
    return *this;
  }
};

//------------------------------------------------------------------------------
template<typename _T>
union Pixel2
{
  typedef _T T;

  struct
  {
    T x,y;
  };
  struct
  {
    T r,g;
  };
  T c[2];

  CUDA_HOST CUDA_DEVICE Pixel2() : x(0), y(0) { }
  CUDA_HOST CUDA_DEVICE Pixel2(T _a) : x(_a), y(_a) { }
  CUDA_HOST CUDA_DEVICE Pixel2(T _x, T _y) : x(_x), y(_y) { }
  CUDA_HOST CUDA_DEVICE ~Pixel2() = default;

  CUDA_HOST CUDA_DEVICE constexpr std::uint8_t numDims() {return 2;}

  CUDA_HOST CUDA_DEVICE operator T() const { return c[0]; }
  CUDA_HOST CUDA_DEVICE T& operator[](size_t i) { return c[i]; }
  CUDA_HOST CUDA_DEVICE const T& operator[](size_t i) const { return c[i]; }
  CUDA_HOST CUDA_DEVICE Pixel2<T>& operator*(const Pixel1<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[0];
    return *this;
  }
  CUDA_HOST CUDA_DEVICE Pixel2<T>& operator*(const Pixel2<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[1];
    return *this;
  }
};

//------------------------------------------------------------------------------
template<typename _T>
union Pixel3
{
  typedef _T T;

  struct
  {
    T x,y,z;
  };
  struct
  {
    T r,g,b;
  };
  T c[3];

  CUDA_HOST CUDA_DEVICE Pixel3() : x(0), y(0), z(0) { }
  CUDA_HOST CUDA_DEVICE Pixel3(T _a) : x(_a), y(_a), z(_a) { }
  CUDA_HOST CUDA_DEVICE Pixel3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) { }
  CUDA_HOST CUDA_DEVICE ~Pixel3() = default;

  CUDA_HOST CUDA_DEVICE constexpr std::uint8_t numDims() {return 3;}

  CUDA_HOST CUDA_DEVICE operator T() const { return c[0]; }
  CUDA_HOST CUDA_DEVICE T& operator[](size_t i) { return c[i]; }
  CUDA_HOST CUDA_DEVICE const T& operator[](size_t i) const { return c[i]; }
  CUDA_HOST CUDA_DEVICE Pixel3<T>& operator*(const Pixel1<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[0];
    c[2] *= rhs[0];
    return *this;
  }
  CUDA_HOST CUDA_DEVICE Pixel3<T>& operator*(const Pixel3<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[1];
    c[2] *= rhs[2];
    return *this;
  }
};

//------------------------------------------------------------------------------
template<typename _T>
union Pixel4
{
  typedef _T T;

  struct
  {
    T x,y,z,w;
  };
  struct
  {
    T r,g,b,a;
  };
  T c[4];

  CUDA_HOST CUDA_DEVICE Pixel4() : x(0), y(0), z(0), w(0) { }
  CUDA_HOST CUDA_DEVICE Pixel4(T _a) : x(_a), y(_a), z(_a), w(_a) { }
  CUDA_HOST CUDA_DEVICE Pixel4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) { }
  CUDA_HOST CUDA_DEVICE ~Pixel4() = default;

  CUDA_HOST CUDA_DEVICE constexpr std::uint8_t numDims() {return 4;}

  CUDA_HOST CUDA_DEVICE operator T() const { return c[0]; }
  CUDA_HOST CUDA_DEVICE T& operator[](size_t i) { return c[i]; }
  CUDA_HOST CUDA_DEVICE const T& operator[](size_t i) const { return c[i]; }
  CUDA_HOST CUDA_DEVICE Pixel4<T>& operator*(const Pixel1<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[0];
    c[2] *= rhs[0];
    c[3] *= rhs[0];
    return *this;
  }
  CUDA_HOST CUDA_DEVICE Pixel4<T>& operator*(const Pixel4<T>& rhs)
  {
    c[0] *= rhs[0];
    c[1] *= rhs[1];
    c[2] *= rhs[2];
    c[3] *= rhs[3];
    return *this;
  }
};

//------------------------------------------------------------------------------
// convenience typedefs
typedef Pixel1<std::uint8_t> Pixel8uC1;
typedef Pixel2<std::uint8_t> Pixel8uC2;
typedef Pixel3<std::uint8_t> Pixel8uC3;
typedef Pixel4<std::uint8_t> Pixel8uC4;

typedef Pixel1<std::uint16_t> Pixel16uC1;
typedef Pixel2<std::uint16_t> Pixel16uC2;
typedef Pixel3<std::uint16_t> Pixel16uC3;
typedef Pixel4<std::uint16_t> Pixel16uC4;

typedef Pixel1<std::int32_t> Pixel32sC1;
typedef Pixel2<std::int32_t> Pixel32sC2;
typedef Pixel3<std::int32_t> Pixel32sC3;
typedef Pixel4<std::int32_t> Pixel32sC4;

typedef Pixel1<float> Pixel32fC1;
typedef Pixel2<float> Pixel32fC2;
typedef Pixel3<float> Pixel32fC3;
typedef Pixel4<float> Pixel32fC4;


//------------------------------------------------------------------------------
// comparison operators
template<typename T>
inline bool operator==(const Pixel1<T>& lhs, const Pixel1<T>& rhs)
{
  return (lhs.x == rhs.x);
}


} // namespace imp

#endif // IMP_PIXEL_HPP

