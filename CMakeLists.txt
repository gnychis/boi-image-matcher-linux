cmake_minimum_required (VERSION 2.6)
project (BOIImageMatcher)

# The version number.
set (MATCHER_VERSION_MAJOR 1)
set (MATCHER_VERSION_MINOR 0)
set (MATCHER_VERSION_PATCH 0)

set (CMAKE_CXX_FLAGS "-std=c++11")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/BOIImageMatcherConfig.h.in"
  "${PROJECT_BINARY_DIR}/BOIImageMatcherConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("~/src/opencv-3.2.0/modules/cudafeatures2d/include")
include_directories("~/src/opencv-3.2.0/modules/cudafilters/include") 
include_directories("~/src/cppzmq-4.2.1")
include_directories("~/src/libzmq-4.2.1/include")

# add the executable
add_executable(boi-image-matcher boi-image-matcher.cpp)
