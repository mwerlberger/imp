cuda_add_executable(cu_rof_denoising_test cu_rof_denoising_test.cpp)
target_link_libraries(cu_rof_denoising_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})

cuda_add_executable(image_pyramid_test image_pyramid_test.cpp)
target_link_libraries(image_pyramid_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})

cuda_add_executable(image_filter_median3x3_test image_filter_median3x3_test.cpp)
target_link_libraries(image_filter_median3x3_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})

cuda_add_executable(image_filter_gauss_test image_filter_gauss_test.cpp)
target_link_libraries(image_filter_gauss_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})
