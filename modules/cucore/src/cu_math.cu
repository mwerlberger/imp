#include <imp/cucore/cu_math.cuh>

#include <imp/core/image.hpp>
#include <imp/cucore/cu_exception.hpp>

namespace imp {
namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
void minMax(const Image<Pixel, pixel_type>& img, Pixel& min, Pixel& max)
{
  if (!img.isGpuMemory())
  {
    // TODO copy to gpu
    throw imp::cu::Exception("CPU memory not yet supported.", __FILE__, __FUNCTION__, __LINE__);
  }

}

template void minMax(const imp::Image8uC1& img, imp::Pixel8uC1& min, imp::Pixel8uC1& max);
template void minMax(const imp::Image8uC2& img, imp::Pixel8uC2& min, imp::Pixel8uC2& max);
template void minMax(const imp::Image8uC3& img, imp::Pixel8uC3& min, imp::Pixel8uC3& max);
template void minMax(const imp::Image8uC4& img, imp::Pixel8uC4& min, imp::Pixel8uC4& max);

template void minMax(const imp::Image16uC1& img, imp::Pixel16uC1& min, imp::Pixel16uC1& max);
template void minMax(const imp::Image16uC2& img, imp::Pixel16uC2& min, imp::Pixel16uC2& max);
template void minMax(const imp::Image16uC3& img, imp::Pixel16uC3& min, imp::Pixel16uC3& max);
template void minMax(const imp::Image16uC4& img, imp::Pixel16uC4& min, imp::Pixel16uC4& max);

template void minMax(const imp::Image32sC1& img, imp::Pixel32sC1& min, imp::Pixel32sC1& max);
template void minMax(const imp::Image32sC2& img, imp::Pixel32sC2& min, imp::Pixel32sC2& max);
template void minMax(const imp::Image32sC3& img, imp::Pixel32sC3& min, imp::Pixel32sC3& max);
template void minMax(const imp::Image32sC4& img, imp::Pixel32sC4& min, imp::Pixel32sC4& max);

template void minMax(const imp::Image32fC1& img, imp::Pixel32fC1& min, imp::Pixel32fC1& max);
template void minMax(const imp::Image32fC2& img, imp::Pixel32fC2& min, imp::Pixel32fC2& max);
template void minMax(const imp::Image32fC3& img, imp::Pixel32fC3& min, imp::Pixel32fC3& max);
template void minMax(const imp::Image32fC4& img, imp::Pixel32fC4& min, imp::Pixel32fC4& max);


} // namespace cu
} // namespace imp
