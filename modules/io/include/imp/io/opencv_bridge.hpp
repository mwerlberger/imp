#ifndef IMP_OPENCV_BRIDGE_HPP
#define IMP_OPENCV_BRIDGE_HPP

#include <memory>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
ImageCvPtr<Pixel,pixel_type>  ocvBridgeLoad(const std::string& filename,
                                            imp::PixelOrder pixel_order)
{
  cv::Mat mat;
  if (pixel_order == PixelOrder::gray)
  {
    mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  }
  else
  {
    // everything else needs color information :)
    mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }

  ImageCvPtr<Pixel,pixel_type> ret;

  switch(pixel_type)
  {
  case imp::PixelType::i8uC1:
    if (pixel_order == PixelOrder::gray)
    {
      ret = std::make_shared<ImageCv<Pixel,pixel_type>>(mat);
    }
    else
    {
      ret = std::make_shared<ImageCv<Pixel,pixel_type>>(mat.cols, mat.rows);
      cv::cvtColor(mat, ret->cvMat(), CV_BGR2GRAY);
    }
    break;
  case imp::PixelType::i32fC1:
     ret = std::make_shared<ImageCv<Pixel,pixel_type>>(mat.cols, mat.rows);
    if (mat.channels() > 1)
    {
      cv::cvtColor(mat, mat, CV_BGR2GRAY);
    }
    mat.convertTo(ret->cvMat(), CV_32F, 1./255.);
    break;
  default:
    throw imp::Exception("Conversion for reading given pixel_type not supported yet.", __FILE__, __FUNCTION__, __LINE__);
  }

  return ret;
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

template<typename Pixel, imp::PixelType pixel_type>
void ocvBridgeShow(const std::string& winname, const ImageCv<Pixel,pixel_type>& img,
                   bool normalize=false)
{
  if (normalize)
  {
    int mat_type = (img.nChannels() > 1) ? CV_8UC3 : CV_8UC1;
    cv::Mat norm_mat(img.height(), img.width(), mat_type);
    cv::normalize(img.cvMat(), norm_mat, 0, 255, CV_MINMAX, CV_8U);
    cv::imshow(winname, norm_mat);
  }
  else
  {
    cv::imshow(winname, img.cvMat());
  }
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

//
// CUDA STUFF
//

namespace cu
{

template<typename Pixel, imp::PixelType pixel_type>
imp::cu::ImageGpuPtr<Pixel,pixel_type> ocvBridgeLoad(const std::string& filename,
                                                     imp::PixelOrder pixel_order)
{
  ImageCvPtr<Pixel,pixel_type> cv_img = imp::ocvBridgeLoad<Pixel, pixel_type>(filename, pixel_order);
  return std::make_shared<imp::cu::ImageGpu<Pixel,pixel_type>>(*cv_img);
}

template<typename Pixel, imp::PixelType pixel_type>
void ocvBridgeShow(const std::string& winname, const ImageGpu<Pixel,pixel_type>& img, bool normalize=false)
{
  const ImageCv<Pixel, pixel_type> cv_img(img);
  imp::ocvBridgeShow(winname, cv_img, normalize);
}

}



} // namespace imp

#endif // IMP_OPENCV_BRIDGE_HPP
