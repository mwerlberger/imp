#ifndef IMP_OPENCV_BRIDGE_HPP
#define IMP_OPENCV_BRIDGE_HPP

#include <memory>

#include <imp/core/image.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/core/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>

namespace imp {

enum class OcvBridgeLoadAs
{
  raw,
  cuda,
  cvmat
};


template<typename Pixel, imp::PixelType pixel_type>
ImageCvPtr<Pixel,pixel_type>  ocvBridgeLoad(const std::string& filename, imp::PixelOrder pixel_order)
{
  switch(pixel_order)
  {
  case PixelOrder::bgr:
  case PixelOrder::undefined:
  return std::make_shared<ImageCv<Pixel,pixel_type>>(cv::imread(filename, CV_LOAD_IMAGE_COLOR));
  default:
  case PixelOrder::gray:
  return std::make_shared<ImageCv<Pixel,pixel_type>>(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
  }
}


template<typename Pixel, imp::PixelType pixel_type>
void ocvBridgeSave(const std::string& filename, const ImageCv<Pixel,pixel_type>& img, bool normalize=false)
{
  if (normalize)
  {
    // TODO
  }
  cv::imwrite(filename, img.cvMat());
}


//template<typename Pixel, imp::PixelType pixel_type>
//std::shared_ptr<> ocv_bridge_imread(const std::string& filename, OcvBridgeLoadAs load_as=OcvBridgeLoadAs::raw)
//{
//  switch (load_as)
//  {
//  case OcvBridgeLoadAs::cvmat:
//    break;
//  case OcvBridgeLoadAs::cuda:
//    break;
//  case OcvBridgeLoadAs::raw:
//  default:
//    break;

//  }
//}


} // namespace imp

#endif // IMP_OPENCV_BRIDGE_HPP
