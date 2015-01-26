add_executable(size_test size_test.cpp)
target_link_libraries(size_test ${IMP_LIBRARIES})

add_executable(roi_test roi_test.cpp)
target_link_libraries(roi_test ${IMP_LIBRARIES})

add_executable(linearhostmemory_test linearhostmemory_test.cpp)
target_link_libraries(linearhostmemory_test ${IMP_LIBRARIES})

find_package( OpenCV REQUIRED core highgui )

add_executable(image_raw_test image_raw_test.cpp)
target_link_libraries(image_raw_test ${IMP_LIBRARIES} ${OpenCV_LIBS})

if (${IMP_WITH_OPENCV})
  # only build these tests when we have opencv connections active

  add_executable(image_cv_test image_cv_test.cpp)
  target_link_libraries(image_cv_test ${IMP_LIBRARIES} ${OpenCV_LIBS})
endif ()
