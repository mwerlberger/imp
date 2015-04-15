cmake_minimum_required(VERSION 2.8.3)
project(imp_cmake)

find_package(catkin_simple REQUIRED)
find_package(catkin REQUIRED)

catkin_package(CFG_EXTRAS imp_cmake-extras.cmake)

# install the included cmake files
install(DIRECTORY cmake/macros DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cmake)
# un-comment this line if we add custom modules
#install(DIRECTORY cmake/modules DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cmake) 