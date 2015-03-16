#include <ros/ros.h>
#include <imp/cuimgproc/cu_rof_denoising.cuh>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "IMP ROF Denoising");
  ros::NodeHandle nh;
  {
    imp::cu::RofDenoising8uC1 rof;

    // subscribe to cam msgs
//    std::string cam_topic(vk::getParam<std::string>("imp/rof/cam_topic", "camera/image_raw"));
//    image_transport::ImageTransport it(nh);
    //image_transport::Subscriber it_sub = it.subscribe(cam_topic, 5, &svo::VoNode::imgCb, &vo_node);

  }

  std::cout << "hmmmm" << std::endl;
  ros::spin();
  return EXIT_SUCCESS;
}
