add_executable(size_test size_test.cpp)
target_link_libraries(size_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS})

add_executable(roi_test roi_test.cpp)
target_link_libraries(roi_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS})

add_executable(linearmemory_test linearmemory_test.cpp)
target_link_libraries(linearmemory_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS})

add_executable(image_raw_test image_raw_test.cpp)
target_link_libraries(image_raw_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})

if (${IMP_WITH_OPENCV})
  # only build these tests when we have opencv connections active
  add_executable(image_cv_test image_cv_test.cpp)
  target_link_libraries(image_cv_test ${IMP_LIBRARIES} ${IMP_LIBRARY_DEPENDS} ${OpenCV_LIBS})
endif ()
