#ifndef IMP_CU_MIN_MAX_IMPL_CU
#define IMP_CU_MIN_MAX_IMPL_CU

#include <imp/core/linearmemory.hpp>
#include <imp/core/image.hpp>
#include <imp/cucore/cu_math.cuh>
#include <imp/cucore/cu_exception.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cucore/cu_texture.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_linearmemory.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
template<typename Pixel>
__global__ void k_minMax(Pixel* d_col_mins, Pixel* d_col_maxs,
                         std::uint32_t roi_width, std::uint32_t roi_height,
                         Texture2D img_tex)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = 0;

  if (x<roi_width)
  {
    float xx = x + 0; // roi_offset TODO
    float yy = y + 0; // roi_offset TODO

    Pixel cur_min, cur_max;
    img_tex.fetch(cur_min, xx, ++yy);
    cur_max = cur_min;

    Pixel val;
    for (; yy<0+roi_height; ++yy)
    {
      img_tex.fetch(val, xx, yy);
      if (val<cur_min) cur_min = val;
      if (val>cur_max) cur_max = val;
    }

    d_col_mins[x] = cur_min;
    d_col_maxs[x] = cur_max;
  }
}


//-----------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void minMax(const std::shared_ptr<ImageGpu<Pixel, pixel_type>>& img,
            Pixel& min_val, Pixel& max_val)
{
//  if (!img->isGpuMemory())
//  {
//    // TODO copy to gpu
//    throw imp::cu::Exception("CPU memory not yet supported.", __FILE__, __FUNCTION__, __LINE__);
//  }

//  const std::shared_ptr<ImageGpu<Pixel,pixel_type>> img(
//        std::dynamic_pointer_cast<ImageGpu<Pixel,pixel_type>>(img));

  if (!img)
  {
    throw Exception("Image can't be intepreted as ImageGpu but isGpuMemory flag is set.",
                    __FILE__, __FUNCTION__, __LINE__);
  }

  std::unique_ptr<Texture2D> img_tex = img->genTexture();
  imp::Roi2u roi = img->roi();
  Fragmentation<512,1,1> frag(roi.width(), 1);

  imp::cu::LinearMemory<Pixel> d_col_mins(roi.width());
  imp::cu::LinearMemory<Pixel> d_col_maxs(roi.width());

  k_minMax
      <<< frag.dimGrid, frag.dimBlock >>> (d_col_mins.data(), d_col_maxs.data(),
                                           roi.width(), roi.height(), *img_tex);

  imp::LinearMemory<Pixel> h_col_mins(roi.width());
  imp::LinearMemory<Pixel> h_col_maxs(roi.width());
  d_col_mins.copyTo(h_col_mins);
  d_col_maxs.copyTo(h_col_maxs);

  min_val = h_col_mins(0);
  max_val = h_col_maxs(0);

  for (auto i=1u; i<roi.width(); ++i)
  {
    min_val = std::min(min_val, h_col_mins(i));
    max_val = std::max(max_val, h_col_maxs(i));
  }

  IMP_CUDA_CHECK();
}

// template instantiations for all our image types
template void minMax(const std::shared_ptr<ImageGpu8uC1>& img, imp::Pixel8uC1& min, imp::Pixel8uC1& max);
template void minMax(const std::shared_ptr<ImageGpu8uC2>& img, imp::Pixel8uC2& min, imp::Pixel8uC2& max);
//template void minMax(const std::shared_ptr<ImageGpu8uC3>& img, imp::Pixel8uC3& min, imp::Pixel8uC3& max);
template void minMax(const std::shared_ptr<ImageGpu8uC4>& img, imp::Pixel8uC4& min, imp::Pixel8uC4& max);

template void minMax(const std::shared_ptr<ImageGpu16uC1>& img, imp::Pixel16uC1& min, imp::Pixel16uC1& max);
template void minMax(const std::shared_ptr<ImageGpu16uC2>& img, imp::Pixel16uC2& min, imp::Pixel16uC2& max);
//template void minMax(const std::shared_ptr<ImageGpu16uC3>& img, imp::Pixel16uC3& min, imp::Pixel16uC3& max);
template void minMax(const std::shared_ptr<ImageGpu16uC4>& img, imp::Pixel16uC4& min, imp::Pixel16uC4& max);

template void minMax(const std::shared_ptr<ImageGpu32sC1>& img, imp::Pixel32sC1& min, imp::Pixel32sC1& max);
template void minMax(const std::shared_ptr<ImageGpu32sC2>& img, imp::Pixel32sC2& min, imp::Pixel32sC2& max);
//template void minMax(const std::shared_ptr<ImageGpu32sC3>& img, imp::Pixel32sC3& min, imp::Pixel32sC3& max);
template void minMax(const std::shared_ptr<ImageGpu32sC4>& img, imp::Pixel32sC4& min, imp::Pixel32sC4& max);

template void minMax(const std::shared_ptr<ImageGpu32fC1>& img, imp::Pixel32fC1& min, imp::Pixel32fC1& max);
template void minMax(const std::shared_ptr<ImageGpu32fC2>& img, imp::Pixel32fC2& min, imp::Pixel32fC2& max);
//template void minMax(const std::shared_ptr<ImageGpu32fC3>& img, imp::Pixel32fC3& min, imp::Pixel32fC3& max);
template void minMax(const std::shared_ptr<ImageGpu32fC4>& img, imp::Pixel32fC4& min, imp::Pixel32fC4& max);


} // namespace cu
} // namespace imp

#endif // IMP_CU_MIN_MAX_IMPL_CU
