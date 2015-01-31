message(STATUS "!!!!!! NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})

cuda_add_executable(cu_linearmemory_test cu_linearmemory_test.cu)
target_link_libraries(cu_linearmemory_test ${IMP_LIBRARIES}  ${IMP_LIBRARY_DEPENDS})

cuda_add_executable(cu_image_gpu_test cu_image_gpu_test.cpp)
target_link_libraries(cu_image_gpu_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})
