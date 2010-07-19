
file(GLOB deepfishermapping_src . "*.cpp")

add_executable(deepfishermapping   ${deepfishermapping_src})

target_link_libraries(deepfishermapping liblearning)