#ifndef IMP_CU_MATH_CUH
#define IMP_CU_MATH_CUH

#include <memory>
#include <imp/cucore/cu_image_gpu.cuh>

namespace imp {
namespace cu {

/**
 * @brief Finding min and max pixel value of given image
 * @note For multi-channel images, the seperate channels are not handeled individually.
 */
template<typename Pixel, imp::PixelType pixel_type>
void minMax(const ImageGpu<Pixel, pixel_type>& img, Pixel& min, Pixel& max);


} // namespace cu
} // namespace imp

#endif // IMP_CU_MATH_CUH

