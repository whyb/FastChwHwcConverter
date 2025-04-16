# FastChwHwcConverter
![CI](https://github.com/whyb/FastChwHwcConverter/workflows/CI/badge.svg)

## Overview
### Multi-Core CPU Implementation (C++Thread OpenMP)
FastChwHwcConverter.hpp is a high-performance, multi-threaded, header-only C++ library for converting image data formats between **HWC (Height, Width, Channels)** and **CHW (Channels, Height, Width)**. It leverages C++ STL Thread / OpenMP for parallel processing, utilizing all CPU cores for maximum performance.

**Note**: If the compilation environment does not find OpenMP, or you set USE_OPENMP to OFF, it will be use C++ thread mode.


### GPU Acceleration (NVIDIA CUDA)
FastChwHwcConverterCuda.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, supporting CUDA versions 10.0+ and above. It requires no installation of the CUDA SDK, header files, or static linking. The library dynamically loads CUDA libraries from the system path. It will automatically search for CUDA's dynamic link library from the system path and dynamically load the functions inside and use them.


**Note**: If your operating environment does not support CUDA or does not meet the conditions for using CUDA acceleration, it will automatically fall back to the CPU (OpenMP/C++ Thread) for processing.
The functions support passing in cuda device memory and host memory parameters.


### GPU Acceleration (AMD ROCm)
FastChwHwcConverterROCm.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, supporting ROCm versions 5.0+ and above. Like the CUDA library, it does not require the ROCm (HIP) SDK, header files, or static linking, and dynamically loads ROCm libraries from the system path.


**Note**: If your operating environment does not support ROCm or does not meet the conditions for using ROCm acceleration, it will automatically fall back to the CPU (OpenMP) for processing.
The functions support passing in ROCm device memory and host memory parameters.


Any similar type conversion code you find another project on GitHub will most likely only achieve performance close to the speed of [single-thread execution](#benchmark-performance-timing-results).

## Table of Contents
- [Overview](#overview)
  - [Multi-Core CPU Implementation (C++Thread OpenMP)](#multi-core-cpu-implementation-cthread-openmp)
  - [GPU Acceleration (NVIDIA CUDA)](#gpu-acceleration-nvidia-cuda)
  - [GPU Acceleration (AMD ROCm)](#gpu-acceleration-amd-rocm)
- [The difference between CHW and HWC](#the-difference-between-chw-and-hwc)
  - [CHW Format](#chw-format)
  - [HWC Format](#hwc-format)
- [Why Convert Between HWC and CHW Formats?](#why-convert-between-hwc-and-chw-formats)
- [Features](#features)
- [Installation](#installation)
  - [for CPU (C++ Thread)](#for-cpu-c-thread)
  - [for CPU (OpenMP)](#for-cpu-openmp)
  - [for GPU (CUDA or ROCm)](#for-gpu-cuda-or-rocm)
- [Requirements](#requirements)
- [API Documents](#api-documents)
  - [HWC -> CHW (CPU)](#hwc-to-chw-conversion-cpu)
  - [CHW -> HWC (CPU)](#chw-to-hwc-conversion-cpu)
  - [HWC -> CHW (CUDA)](#hwc-to-chw-conversion-cuda)
  - [CHW -> HWC (CUDA)](#chw-to-hwc-conversion-cuda)
  - [HWC -> CHW (ROCm)](#hwc-to-chw-conversion-rocm)
  - [CHW -> HWC (ROCm)](#chw-to-hwc-conversion-rocm)
  - [Example](#example)
- [Benchmark Performance Timing Results](#benchmark-performance-timing-results)
- [Contact](#contact)

## The difference between CHW and HWC
Let's consider a 2x2 image with three channels (RGB).
* Example Image Data:
    ```
    Pixel 1 (R, G, B)    Pixel 2 (R, G, B)
    Pixel 3 (R, G, B)    Pixel 4 (R, G, B)
    ```
    We can store this image data in two different formats: CHW (Channel-Height-Width) and HWC (Height-Width-Channel).

### CHW Format
**CHW Format**: In this format, the data is stored channel by channel. First, all the red channel data, then all the green channel data, and finally all the blue channel data.

For example (2x2 RGB Image):
```
RRRRGGGGBBBB
```
Mapping to the actual pixel positions:
```
R1, R2, R3, R4, G1, G2, G3, G4, B1, B2, B3, B4
```
### HWC Format
**HWC Format**: In this format, the data is stored by each pixel's channels in sequence. So, the RGB data for each pixel is stored together.

For example (2x2 RGB Image):
```
RGBRGBRGBRGB
```
Mapping to the actual pixel positions:
```
(R1, G1, B1), (R2, G2, B2), (R3, G3, B3), (R4, G4, B4)
```

## Why Convert Between HWC and CHW Formats?
The conversion between HWC (Height-Width-Channel) and CHW (Channel-Height-Width) formats is crucial for optimizing image processing tasks. Different machine learning frameworks and libraries have varying data format preferences. For instance, many deep learning frameworks, such as PyTorch, prefer the CHW format, while libraries like OpenCV often use the HWC format. By converting between these formats, we ensure compatibility and efficient data handling, enabling seamless transitions between different processing pipelines and maximizing performance for specific tasks. This flexibility enhances the overall efficiency and effectiveness of image processing and machine learning workflows.

## Features
- **High-Performance**: Utilizes C++ Thread / OpenMP for parallel processing. Make full use of CPU multi-core features.
- **GPU Optimization**: Fully leverages NVIDIA CUDA and AMD ROCm technologies to harness the computational power of GPUs, accelerating performance for intensive workloads.
- **Header-Only**: Include **ONLY** a single header file. Easy to integrate into your C/C++ project. [example](#example).
- **Flexible**: Supports scaling, clamping, and normalization of image data, any data type.
- **Lightweight & SDK-Free**: No dependency on any external SDKs like CUDA SDK or HIP SDK. The project requires no additional header files or static library linkage, making it clean and easy to deploy.

## Installation
### for CPU (C++ Thread)
Simply include the header file `FastChwHwcConverter.hpp` in your project:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DUSE_TBB=OFF -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=OFF -DBUILD_ROCM_BENCHMARK=OFF -DBUILD_EXAMPLE=OFF -DBUILD_EXAMPLE_OPENCV=OFF

cmake --build build --config Release
```

### for CPU (OpenMP)

 * Option 1:

    ```shell
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DUSE_TBB=OFF -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=OFF -DBUILD_ROCM_BENCHMARK=OFF -DBUILD_EXAMPLE=OFF -DBUILD_EXAMPLE_OPENCV=OFF

    cmake --build build --config Release
    ```
 * Option 2:

    Simply include the header file `FastChwHwcConverter.hpp` in your project:


### for GPU (CUDA or ROCm)

 * Option 1:

    ```shell
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DUSE_TBB=OFF -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=ON -DBUILD_ROCM_BENCHMARK=ON -DBUILD_EXAMPLE=ON -DBUILD_EXAMPLE_OPENCV=ON

    cmake --build build --config Release
    ```

 * Option 2:

    Simply include the header file `FastChwHwcConverterCuda.hpp` or `FastChwHwcConverterRocm.hpp` in your project:

    ```cpp
    #include "FastChwHwcConverterCuda.hpp"
    ```

    ```cpp
    #include "FastChwHwcConverterROCm.hpp"
    ```

Usually you also need to copy the `nvrtc64_***_0.dll` `nvrtc-builtins64_***` (for Windows CUDA) or `hiprtc****.dll` `hiprtc-builtins****.dll` `amd_comgr_*.dll` `amd_comgr****.dll` (for Windows ROCm)  or `libnvrtc.so` (for Linux CUDA) or `libhiprtc.so` (for Linux ROCm) file in the CUDA/ROCm Runtime SDK to the executable program directory, or set CUDA/ROCm SDK HOME as a system environment variable.

In addition, you need to download and install the latest version of the driver from the [NVIDIA drivers website](https://www.nvidia.com/Download/index.aspx) or [AMD drivers website](https://www.amd.com/en/support). Because this project will dynamically load driver file: `nvcuda.dll` (for Windows CUDA) or `amdhip64_6.dll` (for Windows ROCm) or `libcuda.so` (for Linux CUDA) or `libamdhip64.so` (for Linux ROCm).

## Requirements
* C++17 or later
* OpenMP support (optional, set USE_OPENMP to ON for high performance)
* oneTBB support (optional, set USE_TBB to ON for Intel oneTBB's high performance)
* CMake v3.10 or later (optional)
* OpenCV v4.0 or later (optional, if BUILD_EXAMPLE_OPENCV is ON)
* CUDA 11.2+ driver (optional, if you want to use CUDA acceleration, And NVIDIA GPU's compute capability > 3.5, more details see [here](https://developer.nvidia.com/cuda-gpus). )
* ROCm 5.0+ driver (optional, if you want to use ROCm acceleration, hardware and system requirements see [here](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html). )

## API Documents

### HWC to CHW Conversion (CPU)
The `whyb::cpu::hwc2chw()` function converts image data from HWC format to CHW format.
```cpp
template <typename Stype, typename Dtype,
            bool HasAlpha = false,
            bool NeedClamp = false,
            bool NeedNormalizedMeanStds = false>
void hwc2chw(
    const size_t h, const size_t w, const size_t c,
    const Stype* src, Dtype* dst,
    const Dtype alpha = 1, 
    const Dtype min_v = 0.0, const Dtype max_v = 1.0,
    const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
    const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }
);
```

Parameters:

* `h`: Height of the image.
* `w`: Width of the image.
* `c`: Number of channels.
* `src`: Pointer to the source data in HWC format.
* `dst`: Pointer to the destination data in CHW format.
* `alpha`: Scaling factor (default is 1).
* `min_v`: Minimum value for clamping (default is 0.0).
* `max_v`: Maximum value for clamping (default is 1.0).
* `mean`: Array of mean values for normalization (default is {0.485, 0.456, 0.406}).
* `stds`: Array of standard deviation values for normalization (default is {0.229, 0.224, 0.225}).

### CHW to HWC Conversion (CPU)
The `whyb::cpu::chw2hwc()` function converts image data from CHW format to HWC format.

```cpp
template <typename Stype, typename Dtype,
            bool HasAlpha = false,
            bool NeedClamp = false>
void chw2hwc(
    const size_t c, const size_t h, const size_t w,
    const Stype* src, Dtype* dst, 
    const Dtype alpha = 1, 
    const Dtype min_v = 0, const Dtype max_v = 255
);
```
Parameters:

* `c`: Number of channels.
* `h`: Height of the image.
* `w`: Width of the image.
* `src`: Pointer to the source data in CHW format.
* `dst`: Pointer to the destination data in HWC format.
* `alpha`: Scaling factor (default is 1).
* `min_v`: Minimum value for clamping (default is 0).
* `max_v`: Maximum value for clamping (default is 255).


### HWC to CHW Conversion (CUDA)
The `whyb::nvidia::hwc2chw()` function converts image data from HWC format to CHW format.
```cpp
void hwc2chw(
    const size_t h, const size_t w, const size_t c,
    const uint8_t* src, float* dst,
    const float alpha = 1.f/255.f
);
```

Parameters:

* `h`: Height of the image.
* `w`: Width of the image.
* `c`: Number of channels.
* `src`: Pointer to the source data(host memory) in HWC format.
* `dst`: Pointer to the destination data(host memory) in CHW format.
* `alpha`: Scaling factor (default is 1).

**Note**: Please call whyb::nvidia::init() before the first use, and call whyb::nvidia::release() to release it after confirming that it will not be used anymore.

### CHW to HWC Conversion (CUDA)
The `whyb::nvidia::chw2hwc()` function converts image data from CHW format to HWC format.

```cpp
void chw2hwc(
    const size_t c, const size_t h, const size_t w,
    const float* src, uint8_t* dst,
    const uint8_t alpha = 255.0f
);
```
Parameters:

* `c`: Number of channels.
* `h`: Height of the image.
* `w`: Width of the image.
* `src`: Pointer to the source data(host memory) in CHW format.
* `dst`: Pointer to the destination data(host memory) in HWC format.
* `alpha`: Scaling factor (default is 1).

**Note**: Please call whyb::nvidia::init() before the first use, and call whyb::nvidia::release() to release it after confirming that it will not be used anymore.

### HWC to CHW Conversion (ROCm)
The `whyb::amd::hwc2chw()` function converts image data from HWC format to CHW format.
```cpp
void hwc2chw(
    const size_t h, const size_t w, const size_t c,
    const uint8_t* src, float* dst,
    const float alpha = 1.f/255.f
);
```

Parameters:

* `h`: Height of the image.
* `w`: Width of the image.
* `c`: Number of channels.
* `src`: Pointer to the source data(host memory) in HWC format.
* `dst`: Pointer to the destination data(host memory) in CHW format.
* `alpha`: Scaling factor (default is 1).

**Note**: Please call whyb::amd::init() before the first use, and call whyb::amd::release() to release it after confirming that it will not be used anymore.

### CHW to HWC Conversion (ROCm)
The `whyb::amd::chw2hwc()` function converts image data from CHW format to HWC format.

```cpp
void chw2hwc(
    const size_t c, const size_t h, const size_t w,
    const float* src, uint8_t* dst,
    const uint8_t alpha = 255.0f
);
```
Parameters:

* `c`: Number of channels.
* `h`: Height of the image.
* `w`: Width of the image.
* `src`: Pointer to the source data(host memory) in CHW format.
* `dst`: Pointer to the destination data(host memory) in HWC format.
* `alpha`: Scaling factor (default is 1).

**Note**: Please call whyb::amd::init() before the first use, and call whyb::amd::release() to release it after confirming that it will not be used anymore.

### Example
This example code(**test/example.cpp**) demonstrates how to use the FastChwHwcConverter and FastChwHwcConverterCuda library to convert image data from HWC format to CHW format, and then back to HWC format after AI inference.

```cpp
#include "FastChwHwcConverter.hpp"
#include "FastChwHwcConverterCuda.hpp"
#include "FastChwHwcConverterROCm.hpp"
#include <vector>
#include <cstdint>
#include <iostream>

void cpu_example()
{
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::cpu::hwc2chw<uint8_t, float>(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::cpu::chw2hwc<float, uint8_t>(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    std::cout << "cpu example done" << std::endl;
}

void cuda_example()
{
    if (!whyb::nvidia::init()) { return; }
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::nvidia::hwc2chw(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::nvidia::chw2hwc(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    whyb::nvidia::release();
    std::cout << "cuda example done" << std::endl;
}

void rocm_example()
{
    if (!whyb::amd::init()) { return; }
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::amd::hwc2chw(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f / 255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::amd::chw2hwc(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    whyb::amd::release();
    std::cout << "rocm example done" << std::endl;
}

int main() {
    cpu_example();
    cuda_example();
    rocm_example();
    return 0;
}
```
If you are using OpenCV's `cv::Mat`, please refer to the **test/example-opencv.cpp** file.

## Benchmark Performance Timing Results

The table below shows the benchmark performance timing for different image dimensions, channels, and processing configurations.

    RAM: DDR5 2400MHz 4x32-bit channels
    CPU(OpenMP): Intel(R) Core(TM) i7-13700K
    GPU(CUDA): NVIDIA GeForce RTX 3060 Ti
    GPU(ROCm): AMD Radeon RX 6900 XT

|             |CPU(Single)|CPU(Single)|CPU(OpenMP)|CPU(OpenMP)|   CUDA  |   CUDA  |   ROCm  |   ROCm  |
|-------------|---------|----------|-----------|-----------|---------|---------|---------|---------|
|  W x H x C  | hwc2chw | chw2hwc  | hwc2chw   | chw2hwc   | hwc2chw | chw2hwc | hwc2chw | chw2hwc |
| 426x240x1 | 0.097ms | 0.110ms  | 0.113ms   | 0.030ms   | 0.022ms | 0.019ms | 0.059ms | 0.053ms |
| 426x240x3 | 0.331ms | 0.314ms  | 0.061ms   | 0.068ms   | 0.022ms | 0.019ms | 0.062ms | 0.059ms |
| 426x240x4 | 0.439ms | 0.415ms  | 0.082ms   | 0.082ms   | 0.020ms | 0.019ms | 0.062ms | 0.061ms |
| 640x360x1 | 0.217ms | 0.236ms  | 0.048ms   | 0.052ms   | 0.022ms | 0.021ms | 0.062ms | 0.061ms |
| 640x360x3 | 0.743ms | 0.705ms  | 0.147ms   | 0.140ms   | 0.036ms | 0.021ms | 0.060ms | 0.059ms |
| 640x360x4 | 0.881ms | 0.921ms  | 0.219ms   | 0.203ms   | 0.025ms | 0.021ms | 0.057ms | 0.053ms |
| 854x480x1 | 0.393ms | 0.415ms  | 0.094ms   | 0.089ms   | 0.025ms | 0.024ms | 0.063ms | 0.060ms |
| 854x480x3 | 1.328ms | 1.269ms  | 0.250ms   | 0.232ms   | 0.029ms | 0.024ms | 0.052ms | 0.052ms |
| 854x480x4 | 1.717ms | 1.670ms  | 0.263ms   | 0.262ms   | 0.034ms | 0.027ms | 0.054ms | 0.051ms |
| 1280x720x1 | 0.873ms | 0.937ms  | 0.130ms   | 0.180ms   | 0.053ms | 0.040ms | 0.060ms | 0.052ms |
| 1280x720x3 | 2.877ms | 2.828ms  | 0.449ms   | 0.457ms   | 0.052ms | 0.042ms | 0.061ms | 0.056ms |
| 1280x720x4 | 3.558ms | 3.848ms  | 0.719ms   | 0.616ms   | 0.054ms | 0.045ms | 0.062ms | 0.056ms |
| 1920x1080x1 | 1.949ms | 2.136ms  | 0.374ms   | 0.342ms   | 0.081ms | 0.067ms | 0.079ms | 0.060ms |
| 1920x1080x3 | 6.587ms | 6.469ms  | 1.000ms   | 0.672ms   | 0.087ms | 0.074ms | 0.080ms | 0.064ms |
| 1920x1080x4 | 8.144ms | 8.615ms  | 0.832ms   | 0.914ms   | 0.103ms | 0.080ms | 0.077ms | 0.057ms |
| 2560x1440x1 | 3.530ms | 3.800ms  | 0.423ms   | 0.476ms   | 0.114ms | 0.116ms | 0.094ms | 0.074ms |
| 2560x1440x3 | 11.47ms | 11.611ms | 1.323ms   | 1.169ms   | 0.142ms | 0.127ms | 0.089ms | 0.070ms |
| 2560x1440x4 | 14.14ms | 15.273ms | 2.391ms   | 2.567ms   | 0.154ms | 0.136ms | 0.094ms | 0.075ms |
| 3840x2160x1 | 7.976ms | 8.494ms  | 1.103ms   | 1.387ms   | 0.234ms | 0.227ms | 0.129ms | 0.097ms |
| 3840x2160x3 | 26.30ms | 25.824ms | 5.339ms   | 4.438ms   | 0.307ms | 0.253ms | 0.132ms | 0.096ms |
| 3840x2160x4 | 32.94ms | 34.718ms | 5.805ms   | 4.514ms   | 0.323ms | 0.272ms | 0.131ms | 0.097ms |
| 7680x4320x1 | 31.54ms | 34.100ms | 5.742ms   | 4.976ms   | 0.836ms | 0.741ms | 0.484ms | 0.214ms |
| 7680x4320x3 | 102.87ms| 102.42ms| 19.261ms  | 17.294ms  | 1.057ms | 0.890ms | 0.621ms | 0.222ms |
| 7680x4320x4 | 133.08ms| 136.31ms| 23.398ms  | 18.445ms  | 1.144ms | 1.013ms | 0.686ms | 0.220ms |

## Contact
For any questions or suggestions, please open an issue or contact the me.
