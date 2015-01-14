set(imp_module_description "Core Functionality")
imp_add_module(core)


set(HEADERS
#  include/imp/core.h
  include/imp/core/globaldefs.h
  include/imp/core/coredefs.h
#  include/imp/core/memorydefs.h
  include/imp/core/types.hpp
  include/imp/core/exception.hpp
  include/imp/core/size.hpp
  include/imp/core/roi.hpp
  include/imp/core/pixel_enums.hpp
  include/imp/core/linearmemory_base.hpp
  include/imp/core/linearmemory.hpp
  include/imp/core/image_base.hpp
  include/imp/core/image.hpp
  include/imp/core/image_allocator.hpp
  include/imp/core/image_raw.hpp
  )

set(SOURCES
  src/linearmemory.cpp
  src/image_raw.cpp
#  src/core.cpp
  )


#imp_module_include_directories(${module})
imp_create_module(${SOURCES} ${HEADERS})
