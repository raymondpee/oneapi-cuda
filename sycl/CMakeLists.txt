cmake_minimum_required(VERSION 3.12)
project(sycl_test)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")

file(GLOB SRC_FILES testcases/*.cpp)
add_executable(sycl_test ${SRC_FILES} main.cpp)
target_link_libraries(sycl_test ${GTEST_LIBRARIES} pthread)

add_custom_target(run
    COMMAND SYCL_DEVICE_FILTER=cuda SYCL_PI_TRACE=1 ./sycl_test
    DEPENDS sycl_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}   
)