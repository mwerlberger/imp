# imp

c++ library for image processing

The library will start from the last revision of the imageutilities library. The
goal is to seperate the CUDA functionality and create a seperate (optional)
library with that and, in addition, extend the library with (e.g.) OpenCL
bindings.


# Setup

## Dependencies

- CUDA
- catkin
- catkin_simple
- cv_bridge
- glog
- gtest
- gflags (TODO)

### Command Line Tools Dependencies

- eigen_catkin

### OpenCV Bridge Dependencies

- cv_bridge


### Pangolin Bridge Dependencies

- glew

### Ros Node Dependencies

- roscpp
- std_msgs
- image_transport
- rospy
- dynamic_reconfigure
