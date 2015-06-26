#ifndef IMP_CU_TEXTURE_CUH
#define IMP_CU_TEXTURE_CUH

#include <cuda_runtime.h>
#include <imp/cu_core/cu_texture2d.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
__device__ __forceinline__ void tex2DFetch(
    imp::Pixel8uC1& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  texel = imp::Pixel8uC1(::tex2D<uchar1>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel8uC2& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  uchar2 val = ::tex2D<uchar2>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel8uC2(val.x, val.y);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel8uC4& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  uchar4 val = ::tex2D<uchar4>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel8uC4(val.x, val.y, val.z, val.w);
}

//-----------------------------------------------------------------------------
__device__ __forceinline__ void tex2DFetch(
    imp::Pixel16uC1& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  texel = imp::Pixel16uC1(::tex2D<ushort1>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel16uC2& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  ushort2 val = ::tex2D<ushort2>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel16uC2(val.x, val.y);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel16uC4& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  ushort4 val = ::tex2D<ushort4>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel16uC4(val.x, val.y, val.z, val.w);
}

//-----------------------------------------------------------------------------
__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32sC1& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  texel = imp::Pixel32sC1(::tex2D<int1>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32sC2& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  int2 val = ::tex2D<int2>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32sC2(val.x, val.y);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32sC4& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  int4 val = ::tex2D<int4>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32sC4(val.x, val.y, val.z, val.w);
}

//-----------------------------------------------------------------------------
__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32fC1& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  texel = imp::Pixel32fC1(::tex2D<float1>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32fC2& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  float2 val = ::tex2D<float2>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32fC2(val.x, val.y);
}

__device__ __forceinline__ void tex2DFetch(
    imp::Pixel32fC4& texel, const Texture2D& tex, float x, float y,
    float mul_x=1.f, float mul_y=1.f, float add_x=0.f, float add_y=0.f)
{
  float4 val = ::tex2D<float4>(tex.tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32fC4(val.x, val.y, val.z, val.w);
}


} // namespace cu
} // namespace imp

#endif // IMP_CU_TEXTURE_CUH

