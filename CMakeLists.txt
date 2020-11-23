# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("homography_estimation")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( homography_estimation homography_estimation.cpp )
target_link_libraries( homography_estimation ${OpenCV_LIBS} )

