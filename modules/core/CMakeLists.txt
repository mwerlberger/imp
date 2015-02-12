set(imp_module_description "Core Functionality")
imp_add_module(core)


set(HEADERS
  include/imp/core/types.hpp
  include/imp/core/exception.hpp
  include/imp/core/size.hpp
  include/imp/core/roi.hpp
  include/imp/core/pixel.hpp
  include/imp/core/pixel_enums.hpp
  include/imp/core/linearmemory_base.hpp
  include/imp/core/linearmemory.hpp
  include/imp/core/image_base.hpp
  include/imp/core/image.hpp
  include/imp/core/memory_storage.hpp
  include/imp/core/image_raw.hpp
  )

set(SOURCES
  src/linearmemory.cpp
  src/image_raw.cpp
  )

# add opencv connectors if openc has been found
if (${IMP_WITH_OPENCV})
  message("[+][core] Build with OpenCV connectors.")
  set(HEADERS
     ${HEADERS}
     include/imp/core/cv_connector_pixel_types.hpp
     include/imp/core/image_cv.hpp
  )
  set(SOURCES
     ${SOURCES}
     src/cv_connector_pixel_types.cpp
     src/image_cv.cpp
  )
else()
  message("[-][core] Build without OpenCV connectors.")
endif()

imp_create_module(${SOURCES} ${HEADERS})
