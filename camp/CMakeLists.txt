
file(GLOB camp_src . "*.cpp")

add_definitions(-DCAMP_EXPORTS)

add_library(camp  SHARED  ${camp_src})