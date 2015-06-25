#include <imp/cu_core/cu_texture.cuh>

#include <cuda_runtime.h>

namespace imp {
namespace cu {


__device__ void Texture2D::fetch(
    imp::Pixel8uC1& texel, float x, float y,
    float mul_x, float mul_y, float add_x, float add_y)
{
  texel = imp::Pixel8uC1(tex2D<uchar1>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}
__device__ void Texture2D::fetch(
    imp::Pixel8uC2& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  uchar2 val = tex2D<uchar2>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel8uC2(val.x, val.y);
}
//  __device__ void Texture2D::fetch(imp::Pixel8uC3& texel, float x, float y,
//                                        float mul_x, float mul_y,
//                                        float add_x, float add_y)
//  {
//    uchar3 val = tex2D<uchar3>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
//    texel = imp::Pixel8uC3(val.x, val.y, val.z);
//  }
__device__ void Texture2D::fetch(
    imp::Pixel8uC4& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  uchar4 val = tex2D<uchar4>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel8uC4(val.x, val.y, val.z, val.w);
}

__device__ void Texture2D::fetch(
    imp::Pixel16uC1& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  texel = imp::Pixel16uC1(tex2D<ushort1>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}
__device__ void Texture2D::fetch(
    imp::Pixel16uC2& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  ushort2 val = tex2D<ushort2>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel16uC2(val.x, val.y);
}
//  __device__ void Texture2D::fetch(imp::Pixel16uC3& texel, float x, float y, float mul_x, float mul_y, float add_x, float add_y)
//  {
//    ushort3 val = tex2D<ushort3>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
//    texel = imp::Pixel16uC3(val.x, val.y, val.z);
//  }
__device__ void Texture2D::fetch(
    imp::Pixel16uC4& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  ushort4 val = tex2D<ushort4>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel16uC4(val.x, val.y, val.z, val.w);
}

__device__ void Texture2D::fetch(
    imp::Pixel32sC1& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  texel = imp::Pixel32sC1(tex2D<int1>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}
__device__ void Texture2D::fetch(
    imp::Pixel32sC2& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  int2 val = tex2D<int2>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32sC2(val.x, val.y);
}
//  __device__ void Texture2D::fetch(imp::Pixel32sC3& texel, float x, float y, float mul_x, float mul_y, float add_x, float add_y)
//  {
//    int3 val = tex2D<int3>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
//    texel = imp::Pixel32sC3(val.x, val.y, val.z);
//  }
__device__ void Texture2D::fetch(
    imp::Pixel32sC4& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  int4 val = tex2D<int4>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32sC4(val.x, val.y, val.z, val.w);
}

__device__ void Texture2D::fetch(
    imp::Pixel32fC1& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  texel = imp::Pixel32fC1(tex2D<float1>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f).x);
}
__device__ void Texture2D::fetch(
    imp::Pixel32fC2& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  float2 val = tex2D<float2>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32fC2(val.x, val.y);
}
//  __device__ void Texture2D::fetch(imp::Pixel32fC3& texel, float x, float y, float mul_x, float mul_y, float add_x, float add_y)
//  {
//    float3 val = tex2D<float3>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
//    texel = imp::Pixel32fC3(val.x, val.y, val.z);
//  }
__device__ void Texture2D::fetch(
    imp::Pixel32fC4& texel, float x, float y,
    float mul_x, float mul_y,
    float add_x, float add_y)
{
  float4 val = tex2D<float4>(tex_object, x*mul_x+add_x+0.5f, y*mul_y+add_y+.5f);
  texel = imp::Pixel32fC4(val.x, val.y, val.z, val.w);
}


} // namespace cu
} // namespace imp
