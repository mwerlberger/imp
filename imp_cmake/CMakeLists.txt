cmake_minimum_required(VERSION 2.8.3)
project(imp_cmake)

find_package(catkin_simple REQUIRED)
#find_package(catkin REQUIRED)
catkin_simple()

# install the included cmake files
install(DIRECTORY cmake/macros DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cmake)
# un-comment this line if we add custom modules
install(DIRECTORY cmake/modules DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cmake)

cs_install()
cs_export(CFG_EXTRAS imp_cmake-extras.cmake)

# # copy all macros and modules to the devel space
# message(STATUS "install cmake stuff to: ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}")
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/macros)
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/modules)
