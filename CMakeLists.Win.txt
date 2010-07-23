cmake_minimum_required(VERSION 2.8)

project(liblearning) 

include_directories(include )
include_directories($ENV{INCLUDE})
message(STATUS "include dir: $ENV{INCLUDE}")

link_directories($ENV{LD_LIBRARY_PATH})
message(STATUS "library dir: $ENV{LD_LIBRARY_PATH}")

add_subdirectory(camp)
add_subdirectory(liblearning)
add_subdirectory(deepfishermapping)
add_subdirectory(gendataset)
add_subdirectory(test)


set(library_output_path bin)
set(executable_output_path bin)

