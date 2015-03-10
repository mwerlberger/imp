#include <imp/io/opencv_bridge.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace imp {

template<typename Pixel, imp::PixelType pixel_type>
ImageCv<Pixel,pixel_type>::Ptr ocvBridgeLoad(const std::string& filename, imp::PixelOrder pixel_order)
{
  switch(pixel_order)
  {
  case PixelOrder::gray:
  return std::make_shared<ImageCv<Pixel,pixel_type>>(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
  case PixelOrder::bgr:
  case PixelOrder::undefined:
  return std::make_shared<ImageCv<Pixel,pixel_type>>(cv::imread(filename, CV_LOAD_IMAGE_COLOR));
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


} // namespace imp

