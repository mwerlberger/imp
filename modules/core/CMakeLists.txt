set(imp_module_description "Core Functionality")
imp_add_module(core)


set(HEADERS
  include/imp/core.h
  include/imp/core/globaldefs.h
  include/imp/core/coredefs.h
  include/imp/core/memorydefs.h
  include/imp/core/linearmemory.h
  include/imp/core/linearhostmemory.h
  include/imp/core/setvalue.h
  include/imp/core/copy.h
  )

set(SOURCES
  src/core.cpp
  src/setvalue.cpp
  )


#imp_module_include_directories(${module})
imp_create_module(${SOURCES} ${HEADERS})
