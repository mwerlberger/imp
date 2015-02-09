#include <imp/cucore/cu_math.cuh>

namespace imp {
namespace cu {

template<typename Pixel, imp::PixelType pixel_type>
void minMax(const Image<Pixel, pixel_type>& img, const imp::Roi2u& roi,
            Pixel& min, Pixel& max)
{
//  min = 0;
//  max = 0;
}


} // namespace cu
} // namespace imp
