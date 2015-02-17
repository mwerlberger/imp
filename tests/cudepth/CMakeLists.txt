cuda_add_executable(variational_stereo_test variational_stereo_test.cpp)
target_link_libraries(variational_stereo_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})
