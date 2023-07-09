cmake_minimum_required(VERSION 3.12)
project(simple-sycl-app)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")

add_executable(simple-sycl-app simple-sycl-app.cpp)

add_custom_target(run
    COMMAND SYCL_DEVICE_FILTER=cuda SYCL_PI_TRACE=1 ./simple-sycl-app
    DEPENDS simple-sycl-app
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)