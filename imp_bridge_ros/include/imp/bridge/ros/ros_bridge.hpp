#ifndef IMP_ROS_BRIDGE_HPP
#define IMP_ROS_BRIDGE_HPP

#include <memory>
#include <cstdint>

#include <glog/logging.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

#include <imp/core/exception.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>

namespace imgenc = sensor_msgs::image_encodings;

namespace imp {


std::ostream& operator<<(std::ostream &os, const PixelType& pixel_type)
{
  switch(pixel_type)
  {
  case imp::PixelType::i8uC1:
    os << "imp::PixelType::i8uC1";
  break;
  case imp::PixelType::i8uC2:
    os << "imp::PixelType::i8uC2";
  break;
  case imp::PixelType::i8uC3:
    os << "imp::PixelType::i8uC3";
  break;
  case imp::PixelType::i8uC4:
    os << "imp::PixelType::i8uC4";
  break;

  case imp::PixelType::i16uC1:
    os << "imp::PixelType::i16uC1";
  break;
  case imp::PixelType::i16uC2:
    os << "imp::PixelType::i16uC2";
  break;
  case imp::PixelType::i16uC3:
    os << "imp::PixelType::i16uC3";
  break;
  case imp::PixelType::i16uC4:
    os << "imp::PixelType::i16uC4";
  break;

  case imp::PixelType::i32uC1:
    os << "imp::PixelType::i32uC1";
  break;
  case imp::PixelType::i32uC2:
    os << "imp::PixelType::i32uC2";
  break;
  case imp::PixelType::i32uC3:
    os << "imp::PixelType::i32uC3";
  break;
  case imp::PixelType::i32uC4:
    os << "imp::PixelType::i32uC4";
  break;

  case imp::PixelType::i32sC1:
    os << "imp::PixelType::i32sC1";
  break;
  case imp::PixelType::i32sC2:
    os << "imp::PixelType::i32sC2";
  break;
  case imp::PixelType::i32sC3:
    os << "imp::PixelType::i32sC3";
  break;
  case imp::PixelType::i32sC4:
    os << "imp::PixelType::i32sC4";
  break;
  default:
    os << "unknown PixelType";
  }
  return os;
}

//------------------------------------------------------------------------------
void getPixelTypeFromRosImageEncoding(
    imp::PixelType& pixel_type,
    imp::PixelOrder& pixel_order,
    const std::string& encoding)
{
  //! @todo (MWE) we do not support bayer or YUV images yet.
  if (encoding == imgenc::BGR8)
  {
    pixel_type = imp::PixelType::i8uC3;
    pixel_order = imp::PixelOrder::bgr;
  }
  else if (encoding == imgenc::MONO8)
  {
    pixel_type = imp::PixelType::i8uC1;
    pixel_order = imp::PixelOrder::gray;
  }
  else if (encoding == imgenc::RGB8)
  {
    pixel_type = imp::PixelType::i8uC3;
    pixel_order = imp::PixelOrder::rgb;
  }
  else if (encoding == imgenc::MONO16)
  {
    pixel_type = imp::PixelType::i16uC1;
    pixel_order = imp::PixelOrder::gray;
  }
  else if (encoding == imgenc::BGR16)
  {
    pixel_type = imp::PixelType::i16uC3;
    pixel_order = imp::PixelOrder::bgr;
  }
  else if (encoding == imgenc::RGB16)
  {
    pixel_type = imp::PixelType::i16uC3;
    pixel_order = imp::PixelOrder::rgb;
  }
  else if (encoding == imgenc::BGRA8)
  {
    pixel_type = imp::PixelType::i8uC4;
    pixel_order = imp::PixelOrder::bgra;
  }
  else if (encoding == imgenc::RGBA8)
  {
    pixel_type = imp::PixelType::i8uC4;
    pixel_order = imp::PixelOrder::rgba;
  }
  else if (encoding == imgenc::BGRA16)
  {
    pixel_type = imp::PixelType::i16uC4;
    pixel_order = imp::PixelOrder::bgra;
  }
  else if (encoding == imgenc::RGBA16)
  {
    pixel_type = imp::PixelType::i16uC4;
    pixel_order = imp::PixelOrder::rgba;
  }
  else
  {
    IMP_THROW_EXCEPTION("Unsupported image encoding " + encoding + ".");
  }
}

////------------------------------------------------------------------------------
//template<typename Pixel, imp::PixelType pixel_type>
//void rosBridge(ImageRawPtr<Pixel,pixel_type>& out,
//               const sensor_msgs::ImageConstPtr& img_msg,
//               imp::PixelOrder pixel_order)
//{
//  // TODO
//}


//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void toImageGpu(
    imp::cu::ImageGpuPtr<Pixel, pixel_type>& out,
    const sensor_msgs::Image& src/*,
    imp::PixelOrder pixel_order*/)
{
  imp::PixelType src_pixel_type;
  imp::PixelOrder src_pixel_order;
  imp::getPixelTypeFromRosImageEncoding(src_pixel_type, src_pixel_order, src.encoding);


  int bit_depth = imgenc::bitDepth(src.encoding);
  int num_channels = imgenc::numChannels(src.encoding);
  std::uint32_t width = src.width;
  std::uint32_t height = src.height;
  std::uint32_t pitch = src.step;

  // sanity check
  CHECK_LE(pitch, width * num_channels * bit_depth/8) << "Input image seem to wrongly formatted";

  switch (src_pixel_type)
  {
  case imp::PixelType::i8uC1:
  {
    CHECK_EQ(pixel_type, src_pixel_type) << "src and dst pixel types do not match";
    unsigned char* raw_buffer = const_cast<unsigned char*>(&src.data[0]);
    imp::ImageRaw8uC1 src_wrapped(
          reinterpret_cast<imp::Pixel8uC1*>(&raw_buffer),
        width, height, pitch, true);
    if (!out && out->width() != width && out->height() != height)
    {
      out = std::make_shared<imp::cu::ImageGpu8uC1>(width,height);
    }
    out->copyFrom(src_wrapped);
  }
  break;
//  case imp::PixelType::i8uC2:
//  {
//  }
//  break;
//  case imp::PixelType::i8uC3:
//  {
//  }
//  break;
//  case imp::PixelType::i8uC4:
//  {
//  }
//  break;
//  case imp::PixelType::i16uC1:
//  {
//  }
//  break;
//  case imp::PixelType::i16uC2:
//  {
//  }
//  break;
//  case imp::PixelType::i16uC3:
//  {
//  }
//  break;
//  case imp::PixelType::i16uC4:
//  {
//  }
//  break;
//  case imp::PixelType::i32uC1:
//  {
//  }
//  break;
//  case imp::PixelType::i32uC2:
//  {
//  }
//  break;
//  case imp::PixelType::i32uC3:
//  {
//  }
//  break;
//  case imp::PixelType::i32uC4:
//  {
//  }
//  break;
//  case imp::PixelType::i32sC1:
//  {
//  }
//  break;
//  case imp::PixelType::i32sC2:
//  {
//  }
//  break;
//  case imp::PixelType::i32sC3:
//  {
//  }
//  break;
//  case imp::PixelType::i32sC4:
//  {
//  }
//  break;
//  case imp::PixelType::i32fC1:
//  {
//  }
//  break;
//  case imp::PixelType::i32fC2:
//  {
//  }
//  break;
//  case imp::PixelType::i32fC3:
//  {
//  }
//  break;
//  case imp::PixelType::i32fC4:
//  {

//  }
//  break;
  default:
    IMP_THROW_EXCEPTION("Unsupported pixel type" + src.encoding + ".");
  } // switch(...)

}

} // namespace imp

#endif // IMP_ROS_BRIDGE_HPP
