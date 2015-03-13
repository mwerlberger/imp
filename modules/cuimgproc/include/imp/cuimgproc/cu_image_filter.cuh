#ifndef IMP_CU_IMAGE_FILTER_CUH
#define IMP_CU_IMAGE_FILTER_CUH

#include <imp/core/types.hpp>
#include <imp/core/pixel_enums.hpp>
#include <imp/cucore/cu_image_gpu.cuh>

namespace imp {
namespace cu {

//-----------------------------------------------------------------------------
/** filterMedian3x3 performs a median filter on a 3x3 window
 *
 */
template<typename Pixel, imp::PixelType pixel_type>
void filterMedian3x3(ImageGpu<Pixel, pixel_type>& dst,
                     const ImageGpu<Pixel, pixel_type>& src);

//-----------------------------------------------------------------------------
/** filterGauss performs a gaussian smoothing filter on the given input image \a src
 *
 */
template<typename Pixel, imp::PixelType pixel_type>
void filterGauss(ImageGpu<Pixel, pixel_type>* dst, const ImageGpu<Pixel, pixel_type>* src,
                 float sigma, int kernel_size=0,
                 ImageGpuPtr<Pixel, pixel_type> tmp_img=nullptr);
//                 cudaStream_t stream);

///* ***************************************************************************/
//template<typename Pixel, imp::PixelType pixel_type>
//void cuCubicBSplinePrefilter_32f_C1I(ImageGpu<Pixel, pixel_type>* src);

///* ***************************************************************************/
//// edge filter
//template<typename PixelDst, imp::PixelType pixel_type_dst,
//         typename PixelSrc, imp::PixelType pixel_type_src>
//void filterEdge(ImageGpu<PixelDst, pixel_type_dst>* dst,
//                ImageGpu<PixelSrc, pixel_type_src>* src);

//// edge filter  + evaluation
//template<typename PixelDst, imp::PixelType pixel_type_dst,
//         typename PixelSrc, imp::PixelType pixel_type_src>
//void filterEdge(ImageGpu<PixelDst, pixel_type_dst>* dst,
//                ImageGpu<PixelSrc, pixel_type_src>* src,
//                float alpha, float beta, float minval);

////// edge filter  + evaluation
////void filterEdge(const ImageGpu<Pixel, pixel_type>* src, iu::ImageGpu_32f_C2* dst,
////                float alpha, float beta, float minval);

////// edge filter  + evaluation
////void filterEdge(const ImageGpu<Pixel, pixel_type>* src, iu::ImageGpu_32f_C4* dst,
////                float alpha, float beta, float minval);

////// edge filter  + evaluation
////void filterEdge(const iu::ImageGpu_32f_C4* src, ImageGpu<Pixel, pixel_type>* dst,
////                float alpha, float beta, float minval);

////// edge filter  + evaluation
////void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst,
////                float alpha, float beta, float minval);

////// edge filter  + evaluation
////void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
////                float alpha, float beta, float minval);

///* ***************************************************************************/
//// bilateral filter
//void filterBilateral(ImageGpu<Pixel, pixel_type>* dst,
//                     ImageGpu<Pixel, pixel_type>* src,
//                     const ImageGpu<Pixel, pixel_type>* prior,
//                     const int iters,
//                     const float sigma_spatial, const float sigma_range,
//                     const int radius);

////void filterBilateral(ImageGpu<Pixel, pixel_type>* dst, ImageGpu<Pixel, pixel_type>* src,
////                     const iu::ImageGpu_32f_C4* prior, const int iters,
////                     const float sigma_spatial, const float sigma_range,
////                     const int radius);

////void filterBilateral(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
////                     const iu::ImageGpu_32f_C4* prior, const int iters,
////                     const float sigma_spatial, const float sigma_range,
////                     const int radius);


} // namespace cu
} // namespace imp

#endif // IMP_CU_IMAGE_FILTER_CUH
