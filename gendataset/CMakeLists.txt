
file(GLOB gendataset_src . "*.cpp")

add_executable(gendataset  ${gendataset_src})

target_link_libraries(gendataset liblearning)