cmake_minimum_required(VERSION 3.10)

# Project name
project(FastChwHwcConverter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Apply OpenMP compile options to specific targets
    set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
else()
    set(OpenMP_FLAGS "")
endif()

# Add executable targets
add_executable(example FastChwHwcConverter.hpp example.cpp)
add_executable(benchmark FastChwHwcConverter.hpp benchmark.cpp)

# Include directories for targets
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set compile options for targets
target_compile_options(example PRIVATE ${OpenMP_FLAGS})
target_compile_options(benchmark PRIVATE ${OpenMP_FLAGS})
