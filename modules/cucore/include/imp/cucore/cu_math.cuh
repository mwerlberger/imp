#ifndef IMP_CU_MATH_CUH
#define IMP_CU_MATH_CUH

#include <memory>
#include <imp/core/image.hpp>
#include <imp/core/roi.hpp>

namespace imp {
namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
void minMax(const std::shared_ptr<Image<Pixel, pixel_type>>& img, Pixel& min, Pixel& max);


} // namespace cu
} // namespace imp

#endif // IMP_CU_MATH_CUH

