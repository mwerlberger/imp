#include <imp/core/linearmemory.hpp>
#include <imp/core/image.hpp>
#include <imp/cu_core/cu_math.cuh>
#include <imp/cu_core/cu_exception.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_texture.cuh>
#include <imp/cu_core/cu_utils.hpp>
#include <imp/cu_core/cu_linearmemory.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_minMax(Pixel* d_col_mins, Pixel* d_col_maxs,
                         std::uint32_t roi_x, std::uint32_t roi_y,
                         std::uint32_t roi_width, std::uint32_t roi_height,
                         Texture2D img_tex)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;

  if (x<roi_width)
  {
    float xx = x + roi_x;
    float yy = roi_y;

    Pixel cur_min, cur_max;
    tex2DFetch(cur_min, img_tex, xx, yy++);
    cur_max = cur_min;

    Pixel val;
    for (; yy<roi_y+roi_height; ++yy)
    {
      tex2DFetch(val, img_tex, xx, yy);
      if (val<cur_min) cur_min = val;
      if (val>cur_max) cur_max = val;
    }

    d_col_mins[x] = cur_min;
    d_col_maxs[x] = cur_max;
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel, typename SrcPixel>
__global__ void k_minMax(Pixel* d_col_mins, Pixel* d_col_maxs,
                         SrcPixel* src, size_type src_stride,
                         std::uint32_t roi_x, std::uint32_t roi_y,
                         std::uint32_t roi_width, std::uint32_t roi_height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;

  if (x<roi_width)
  {
    int xx = x+roi_x;
    int yy = roi_y;


    Pixel cur_min, cur_max;
    Pixel val = (Pixel)src[yy++*src_stride+xx];
    cur_min = val;
    cur_max = val;
    for (; yy<roi_y+roi_height; ++yy)
    {
      val = (Pixel)src[yy*src_stride+xx];
      cur_min = imp::cu::min(cur_min, val);
      cur_max = imp::cu::max(cur_max, val);
    }

    d_col_mins[x] = cur_min;
    d_col_maxs[x] = cur_max;
  }
}

//-----------------------------------------------------------------------------
template<typename Pixel>
void minMax(const Texture2D& img_tex, Pixel& min_val, Pixel& max_val, const imp::Roi2u& roi)
{
  Fragmentation<512,1> frag(roi.width(), 1);

  imp::cu::LinearMemory<Pixel> d_col_mins(roi.width());
  imp::cu::LinearMemory<Pixel> d_col_maxs(roi.width());
  IMP_CUDA_CHECK();
  d_col_mins.setValue(Pixel(0));
  d_col_maxs.setValue(Pixel(0));

  k_minMax
      <<<
        frag.dimGrid, frag.dimBlock
      >>> (d_col_mins.data(), d_col_maxs.data(),
           roi.x(), roi.y(), roi.width(), roi.height(), img_tex);
  IMP_CUDA_CHECK();

  imp::LinearMemory<Pixel> h_col_mins(d_col_mins.length());
  imp::LinearMemory<Pixel> h_col_maxs(d_col_maxs.length());
  h_col_mins.setValue(Pixel(0));
  h_col_maxs.setValue(Pixel(0));

  d_col_mins.copyTo(h_col_mins);
  d_col_maxs.copyTo(h_col_maxs);

  min_val = h_col_mins(0);
  max_val = h_col_maxs(0);

  for (auto i=1u; i<roi.width(); ++i)
  {
    min_val = imp::cu::min(min_val, h_col_mins(i));
    max_val = imp::cu::max(max_val, h_col_maxs(i));
  }

  IMP_CUDA_CHECK();
}

//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void minMax(const ImageGpu<Pixel, pixel_type>& img, Pixel& min_val, Pixel& max_val)
{
  IMP_CUDA_CHECK();
  imp::Roi2u roi = img.roi();

#if 0
  cudaResourceDesc tex_res;
  std::memset(&tex_res, 0, sizeof(tex_res));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = img.width();
  tex_res.res.pitch2D.height = img.height();
  tex_res.res.pitch2D.pitchInBytes = img.pitch();
  const void* data = img.cuData();
  tex_res.res.pitch2D.devPtr = const_cast<void*>(data);
  tex_res.res.pitch2D.desc = img.channelFormatDesc();

  cudaTextureDesc tex_desc;
  std::memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModePoint;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex_object;
  cudaError_t err = cudaCreateTextureObject(&tex_object, &tex_res, &tex_desc, 0);
  if  (err != ::cudaSuccess)
  {
    throw imp::cu::Exception("Failed to create texture object", err,
                             __FILE__, __FUNCTION__, __LINE__);
  }
  IMP_CUDA_CHECK();
#endif

#if 0
  std::shared_ptr<Texture2D> img_tex = img.genTexture(
        false, cudaFilterModePoint, cudaAddressModeClamp, cudaReadModeElementType);
  IMP_CUDA_CHECK();
#endif

#if 0
  Texture2D img_tex = imp::cu::bindTexture2D(img, false, cudaFilterModePoint, cudaAddressModeClamp,
                                             cudaReadModeElementType);
  IMP_CUDA_CHECK();
#endif

#if 0
  std::shared_ptr<Texture2D> img_tex = img.genTexture();
  IMP_CUDA_CHECK();
  imp::cu::minMax(*img_tex, min_val, max_val, roi);
  IMP_CUDA_CHECK();
#endif


#if 0
  imp::cu::LinearMemory<Pixel> d_col_mins(roi.width());
  imp::cu::LinearMemory<Pixel> d_col_maxs(roi.width());
  IMP_CUDA_CHECK();
  d_col_mins.setValue(Pixel(0));
  d_col_maxs.setValue(Pixel(0));

  Fragmentation<512,1> frag(roi.width(), 1);
  k_minMax
      <<<
        frag.dimGrid, frag.dimBlock
      >>> (d_col_mins.data(), d_col_maxs.data(),
           img.data(), img.stride(),
           roi.x(), roi.y(), roi.width(), roi.height());
  IMP_CUDA_CHECK();

  imp::LinearMemory<Pixel> h_col_mins(d_col_mins.length());
  imp::LinearMemory<Pixel> h_col_maxs(d_col_maxs.length());
  h_col_mins.setValue(Pixel(0));
  h_col_maxs.setValue(Pixel(0));

  d_col_mins.copyTo(h_col_mins);
  d_col_maxs.copyTo(h_col_maxs);

  min_val = h_col_mins(0);
  max_val = h_col_maxs(0);

  for (auto i=1u; i<roi.width(); ++i)
  {
    min_val = imp::cu::min(min_val, h_col_mins(i));
    max_val = imp::cu::max(max_val, h_col_maxs(i));
  }

  IMP_CUDA_CHECK();
#endif
}


// template instantiations for all our image types
template void minMax(const ImageGpu8uC1& img, imp::Pixel8uC1& min, imp::Pixel8uC1& max);
template void minMax(const ImageGpu8uC2& img, imp::Pixel8uC2& min, imp::Pixel8uC2& max);
//template void minMax(const ImageGpu8uC3& img, imp::Pixel8uC3& min, imp::Pixel8uC3& max);
template void minMax(const ImageGpu8uC4& img, imp::Pixel8uC4& min, imp::Pixel8uC4& max);

template void minMax(const ImageGpu16uC1& img, imp::Pixel16uC1& min, imp::Pixel16uC1& max);
template void minMax(const ImageGpu16uC2& img, imp::Pixel16uC2& min, imp::Pixel16uC2& max);
//template void minMax(const ImageGpu16uC3& img, imp::Pixel16uC3& min, imp::Pixel16uC3& max);
template void minMax(const ImageGpu16uC4& img, imp::Pixel16uC4& min, imp::Pixel16uC4& max);

template void minMax(const ImageGpu32sC1& img, imp::Pixel32sC1& min, imp::Pixel32sC1& max);
template void minMax(const ImageGpu32sC2& img, imp::Pixel32sC2& min, imp::Pixel32sC2& max);
//template void minMax(const ImageGpu32sC3& img, imp::Pixel32sC3& min, imp::Pixel32sC3& max);
template void minMax(const ImageGpu32sC4& img, imp::Pixel32sC4& min, imp::Pixel32sC4& max);

template void minMax(const ImageGpu32fC1& img, imp::Pixel32fC1& min, imp::Pixel32fC1& max);
template void minMax(const ImageGpu32fC2& img, imp::Pixel32fC2& min, imp::Pixel32fC2& max);
//template void minMax(const ImageGpu32fC3& img, imp::Pixel32fC3& min, imp::Pixel32fC3& max);
template void minMax(const ImageGpu32fC4& img, imp::Pixel32fC4& min, imp::Pixel32fC4& max);

} // namespace cu
} // namespace imp
