#pragma once

#include <memory>
#include <pangolin/image_load.h>
#include <imp/core/image_raw.hpp>

namespace imp
{

//------------------------------------------------------------------------------
//template<typename Pixel, imp::PixelType pixel_type>
void pangolinBridgeLoad(std::shared_ptr<imp::ImageRaw8uC1>& out,
                        const std::string& filename, imp::PixelOrder pixel_order)
{
  // try to load an image with pangolin first
  pangolin::TypedImage im = pangolin::LoadImage(
        filename, pangolin::ImageFileType::ImageFileTypePng);

  //! @todo (MWE) FIX input output channel formatting, etc.
  out = std::make_shared<imp::ImageRaw8uC1>((unsigned char*)im.ptr, im.w, im.h, im.pitch, false);
//  switch (im.fmt.channels)
//  {
//  case 1:
//    out = std::make_shared<imp::ImageRaw<Pixel, pixel_type>>(im.w, im.h);
//    break;
//  case 3:
//    out = std::make_shared<imp::ImageRaw<Pixel, pixel_type>>(im.w, im.h);
//    break;
//  case 4:
//    out = std::make_shared<imp::ImageRaw<Pixel, pixel_type>>(im.w, im.h);
//    break;
//  default:
//    throw imp::Exception("Conversion for reading given pixel_type not supported yet.", __FILE__, __FUNCTION__, __LINE__);

//  }
}


} // namespace imp
