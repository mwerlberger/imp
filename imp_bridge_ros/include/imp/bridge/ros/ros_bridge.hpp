#ifndef IMP_ROS_BRIDGE_HPP
#define IMP_ROS_BRIDGE_HPP

#include <memory>

#include <sensor_msgs/Image.h>

namespace imp {

//------------------------------------------------------------------------------
template<typename Pixel, imp::PixelType pixel_type>
void rosBridge(ImageRawPtr<Pixel,pixel_type>& out,
               const sensor_msgs::ImageConstPtr &img_msg,
               imp::PixelOrder pixel_order)
{
  // TODO
}


} // namespace imp

#endif // IMP_ROS_BRIDGE_HPP
