project(impTests CXX C)
find_package(imp)
include_directories("${IMP_INCLUDE_DIRS}")

add_subdirectory(core)
