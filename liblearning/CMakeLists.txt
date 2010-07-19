
file(GLOB_RECURSE liblearning_src . "*.cpp")

add_library(liblearning  STATIC ${liblearning_src})

target_link_libraries(liblearning camp)