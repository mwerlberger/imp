#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <imp/cuimgproc/cu_rof_denoising.cuh>
#include <imp/io/opencv_bridge.hpp>

#include <sensor_msgs/Image.h>

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(30);
  }
  catch (std::exception& e)
  {
    ROS_ERROR("Could not process existing input image. Exception: %s", e.what());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");

  return EXIT_SUCCESS;
}


//int main(int argc, char **argv)
//{
//  ros::init(argc, argv, "IMP ROF Denoising");
//  ros::NodeHandle nh;
//  {
//    imp::cu::RofDenoising8uC1 rof;

//    // subscribe to cam msgs
////    std::string cam_topic(vk::getParam<std::string>("imp/rof/cam_topic", "camera/image_raw"));
////    image_transport::ImageTransport it(nh);
//    //image_transport::Subscriber it_sub = it.subscribe(cam_topic, 5, &svo::VoNode::imgCb, &vo_node);

//  }

//  std::cout << "hmmmm" << std::endl;
//  ros::spin();
//  return EXIT_SUCCESS;
//}
