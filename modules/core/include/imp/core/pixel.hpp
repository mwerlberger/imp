#ifndef IMP_PIXEL_HPP
#define IMP_PIXEL_HPP

#include <cstdint>

namespace imp {

template<typename T>
struct PixelTest
{
  T x;
  PixelTest(T _x) : x(_x) {}
};
typedef PixelTest<std::uint8_t> PixelTest8uC1;

//------------------------------------------------------------------------------
template<typename T>
union Pixel1
{
   struct
   {
      T x;
   };
   struct
   {
      T r;
   };
   T c[1];

   //__device__ operator T () const { return c[0]; }

   Pixel1(T _x) : x(_x) { }
   ~Pixel1() = default;

};

//------------------------------------------------------------------------------
template<typename T>
union Pixel2
{
   struct
   {
      T x,y;
   };
   struct
   {
      T r,g;
   };
   T c[2];

   Pixel2(T _a) : x(_a), y(_a) { }
   Pixel2(T _x, T _y) : x(_x), y(_y) { }
   ~Pixel2() = default;
};

//------------------------------------------------------------------------------
template<typename T>
union Pixel3
{
   struct
   {
      T x,y,z;
   };
   struct
   {
      T r,g,b;
   };
   T c[3];

   Pixel3(T _a) : x(_a), y(_a), z(_a) { }
   Pixel3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) { }
   ~Pixel3() = default;
};

//------------------------------------------------------------------------------
template<typename T>
union Pixel4
{
   struct
   {
      T x,y,z,w;
   };
   struct
   {
      T r,g,b,a;
   };
   T c[4];

   Pixel4(T _a) : x(_a), y(_a), z(_a), w(_a) { }
   Pixel4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) { }
   ~Pixel4() = default;
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

