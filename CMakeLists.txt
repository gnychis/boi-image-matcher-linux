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
file(GLOB SOURCES "*.cpp")
add_executable(boi-image-matcher ${SOURCES})

link_directories("~/src/opencv-3.2.0/build/lib")
target_link_libraries(boi-image-matcher zmq pthread opencv_highgui opencv_imgcodecs opencv_aruco opencv_bgsegm opencv_bioinspired opencv_calib3d opencv_ccalib opencv_core  opencv_datasets  opencv_dnn  opencv_dpm  opencv_face  opencv_features2d  opencv_flann  opencv_freetype  opencv_fuzzy  opencv_highgui  opencv_imgcodecs  opencv_imgproc  opencv_line_descriptor  opencv_ml  opencv_objdetect  opencv_optflow  opencv_phase_unwrapping  opencv_photo  opencv_plot  opencv_reg  opencv_rgbd  opencv_saliency  opencv_shape  opencv_stereo  opencv_stitching  opencv_structured_light  opencv_superres  opencv_surface_matching  opencv_text  opencv_tracking  opencv_videoio  opencv_video  opencv_videostab  opencv_xfeatures2d  opencv_ximgproc  opencv_xobjdetect)
