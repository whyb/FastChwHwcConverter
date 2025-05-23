cmake_minimum_required(VERSION 3.10)

# Project name
project(FastChwHwcConverter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CMake option to control OpenMP usage 
option(USE_OPENMP "Use OpenMP for parallel programming" OFF)

# CMake options to control project generation
option(BUILD_EXAMPLE "Build example target" ON)
option(BUILD_BENCHMARK "Build benchmark target" ON)
option(BUILD_CUDA_BENCHMARK "Build NVIDIA CUDA benchmark target" ON)
option(BUILD_ROCM_BENCHMARK "Build AMD ROCm benchmark target" ON)
option(BUILD_EXAMPLE_OPENCV "Build example-opencv target" OFF)
option(USE_TBB "Use Intel oneTBB for parallel programming" OFF)

# Check for OpenMP support if the option is enabled
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
        add_compile_definitions(USE_OPENMP)
    else()
        set(OpenMP_FLAGS "")
    endif()
else()
    set(OpenMP_FLAGS "")
endif()

# Check for Intel oneTBB support if the option is enabled
if(USE_TBB)
    find_package(TBB REQUIRED)
    message(STATUS "Found Intel oneTBB")
    include_directories(${TBB_INCLUDE_DIRS})
    add_compile_definitions(USE_TBB)
    set(TBB_LIBS TBB::tbb)
else()
    set(TBB_LIBS "")
endif()

# Find OpenCV library
if(BUILD_EXAMPLE_OPENCV)
    find_package(OpenCV REQUIRED)
endif()

# Add executable targets based on options
if(BUILD_EXAMPLE)
    add_executable(example FastChwHwcConverter.hpp test/example.cpp)
    target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TBB_INCLUDE_DIRS})
    target_compile_options(example PRIVATE ${OpenMP_FLAGS})
    target_link_libraries(example PRIVATE ${TBB_LIBS})
endif()

if(BUILD_BENCHMARK)
    add_executable(benchmark FastChwHwcConverter.hpp test/benchmark.cpp)
    target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TBB_INCLUDE_DIRS})
    target_compile_options(benchmark PRIVATE ${OpenMP_FLAGS})
    target_link_libraries(benchmark PRIVATE ${TBB_LIBS})
endif()

if(BUILD_CUDA_BENCHMARK)
    add_executable(cuda_benchmark DynamicLibraryManager.hpp FastChwHwcConverterCuda.hpp test/cuda_benchmark.cpp)
    target_include_directories(cuda_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TBB_INCLUDE_DIRS})
    target_link_libraries(cuda_benchmark PRIVATE ${TBB_LIBS})
endif()

if(BUILD_ROCM_BENCHMARK)
    add_executable(rocm_benchmark DynamicLibraryManager.hpp FastChwHwcConverterROCm.hpp test/rocm_benchmark.cpp)
    target_include_directories(rocm_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TBB_INCLUDE_DIRS})
    target_link_libraries(rocm_benchmark PRIVATE ${TBB_LIBS})
endif()

if(BUILD_EXAMPLE_OPENCV)
    add_executable(example-opencv FastChwHwcConverter.hpp test/example-opencv.cpp)
    target_include_directories(example-opencv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
    target_link_libraries(example-opencv PRIVATE ${OpenCV_LIBS})
    target_compile_options(example-opencv PRIVATE ${OpenMP_FLAGS})
    target_link_libraries(example-opencv PRIVATE ${TBB_LIBS})
endif()
