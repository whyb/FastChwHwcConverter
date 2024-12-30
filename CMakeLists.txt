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
endif()

# Add the executable
add_executable(example FastChwHwcConverter.hpp example.cpp)

# Include directories
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

