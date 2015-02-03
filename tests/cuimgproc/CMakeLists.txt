cuda_add_executable(cu_rof_denoising_test cu_rof_denoising_test.cpp)
target_link_libraries(cu_rof_denoising_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})
