# FastChwHwcConverter
![CI](https://github.com/whyb/FastChwHwcConverter/workflows/CI/badge.svg)

## Overview
### for CPU (OpenMP)
FastChwHwcConverter.hpp is a high-performance, multi-thread, header-only C++ library for converting image data formats between **HWC (Height, Width, Channels)** and **CHW (Channels, Height, Width)**. The library leverages OpenMP parallel processing to provide lightning-fast performance.
### for GPU (CUDA)
FastChwHwcConverterCuda.hpp is a high-performance, fully GPU hardware accelerated, C++ library for converting image data formats between **HWC (Height, Width, Channels)** and **CHW (Channels, Height, Width)**. It supports any version above CUDA 10. The compilation environment does not need to install CUDA SDK, does not need to reference any CUDA header files, and does not need to static link any external dll&so libraries. It will automatically search for CUDA's dynamic link library from the system path and dynamically load the functions inside and use them.


Any similar type conversion code you find another project on GitHub will most likely only achieve performance close to the speed of [single-thread execution](#benchmark-performance-timing-results).

## Table of Contents
- [Overview](#overview)
- [The difference between CHW and HWC](#the-difference-between-chw-and-hwc)
  - [CHW Format](#chw-format)
  - [HWC Format](#hwc-format)
- [Why Convert Between HWC and CHW Formats?](#why-convert-between-hwc-and-chw-formats)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Let's Converter](#lets-converter)
  - [HWC to CHW Conversion (CPU)](#hwc-to-chw-conversion-cpu)
  - [CHW to HWC Conversion (CPU)](#chw-to-hwc-conversion-cpu)
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
- **High-Performance**: Utilizes OpenMP for parallel processing. Make full use of CPU multi-core features.
- **Header-Only**: Include **ONLY** a single header file. Easy to integrate into your C/C++ project. [example](#example).
- **Flexible**: Supports scaling, clamping, and normalization of image data, any data type.

## Installation
### for CPU (OpenMP)
Simply include the header file `FastChwHwcConverter.hpp` in your project:

```cpp
#include "FastChwHwcConverter.hpp"
```

### for GPU (CUDA)
Simply include the header file `FastChwHwcConverterCuda.hpp` in your project:

```cpp
#include "FastChwHwcConverterCuda.hpp"
```

Usually you also need to copy a `nvrtc64_***_0.dll` (for Windows) or `libnvrtc.so`(for Linux) file in the CUDA Runtime SDK to the executable program directory, or set CUDA SDK HOME as a system environment variable.

In addition, you need to download and install the latest version of the driver from the NVIDIA official website (recommended). Because this project will dynamically load driver file: `nvcuda.dll` (for Windows) or `libcuda.so`(for Linux).

## Requirements
* C++11 or later
* OpenMP support (optional, set USE_OPENMP to ON for high performance)
* CMake v3.10 or later (optional)
* OpenCV v4.0 or later (optional, if BUILD_EXAMPLE_OPENCV is ON)
* CUDA 10+(optional, if you want to use cuda acceleration, And has NVIDIA GPU)

## Let's Converter

### HWC to CHW Conversion (CPU)
The `hwc2chw` function converts image data from HWC format to CHW format.
```cpp
template <typename Stype, typename Dtype>
void hwc2chw(
    const size_t h, const size_t w, const size_t c,
    const Stype* src, Dtype* dst,
    const Dtype alpha = 1, 
    const bool clamp = false, const Dtype min_v = 0.0, const Dtype max_v = 1.0,
    const bool normalized_mean_stds = false,
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
* `clamp`: Whether to clamp the data values (default is false).
* `min_v`: Minimum value for clamping (default is 0.0).
* `max_v`: Maximum value for clamping (default is 1.0).
* `normalized_mean_stds`: Whether to use mean and standard deviation for normalization (default is false).
* `mean`: Array of mean values for normalization (default is {0.485, 0.456, 0.406}).
* `stds`: Array of standard deviation values for normalization (default is {0.229, 0.224, 0.225}).

### CHW to HWC Conversion (CPU)
The `chw2hwc` function converts image data from CHW format to HWC format.

```cpp
template <typename Stype, typename Dtype>
void chw2hwc(
    const size_t c, const size_t h, const size_t w,
    const Stype* src, Dtype* dst, 
    const Dtype alpha = 1, 
    const bool clamp = false, const Dtype min_v = 0, const Dtype max_v = 255
);
```
Parameters:

* `c`: Number of channels.
* `h`: Height of the image.
* `w`: Width of the image.
* `src`: Pointer to the source data in CHW format.
* `dst`: Pointer to the destination data in HWC format.
* `alpha`: Scaling factor (default is 1).
* `clamp`: Whether to clamp the data values (default is false).
* `min_v`: Minimum value for clamping (default is 0).
* `max_v`: Maximum value for clamping (default is 255).


### Example
This example code(**test/example.cpp**) demonstrates how to use the FastChwHwcConverter library to convert image data from HWC format to CHW format, and then back to HWC format after AI inference.

```cpp
#include "FastChwHwcConverter.hpp"
#include <vector>

int main() {
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
    whyb::hwc2chw<uint8_t, float>(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data());

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::chw2hwc<float, uint8_t>(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data());

    return 0;
}
```
If you are using OpenCV's `cv::Mat`, please refer to the **test/example-opencv.cpp** file.

## Benchmark Performance Timing Results

The table below shows the benchmark performance timing for different image dimensions, channels, and processing configurations.

CPU: Intel(R) Core(TM) i7-13700K

RAM: DDR5 2400MHz 4x32-bit channels

GPU: NVIDIA GeForce RTX 3060 Ti

|       |        |         | single-thread | single-thread | multi-thread | multi-thread |   CUDA   |   CUDA   |
|-------|--------|---------|---------------|---------------|--------------|--------------|----------|----------|
| Width | Height | Channel | hwc2chw       | chw2hwc       | hwc2chw      | chw2hwc      | hwc2chw  | chw2hwc  |
| 426   | 240    | 1       | 0.097ms       | 0.110ms       | 0.113ms      | 0.030ms      | 0.006ms  | 0.006ms  |
| 426   | 240    | 3       | 0.331ms       | 0.314ms       | 0.061ms      | 0.068ms      | 0.008ms  | 0.007ms  |
| 426   | 240    | 4       | 0.439ms       | 0.415ms       | 0.082ms      | 0.082ms      | 0.010ms  | 0.010ms  |
| 640   | 360    | 1       | 0.217ms       | 0.236ms       | 0.048ms      | 0.052ms      | 0.010ms  | 0.010ms  |
| 640   | 360    | 3       | 0.743ms       | 0.705ms       | 0.147ms      | 0.140ms      | 0.013ms  | 0.015ms  |
| 640   | 360    | 4       | 0.881ms       | 0.921ms       | 0.219ms      | 0.203ms      | 0.015ms  | 0.016ms  |
| 854   | 480    | 1       | 0.393ms       | 0.415ms       | 0.094ms      | 0.089ms      | 0.009ms  | 0.008ms  |
| 854   | 480    | 3       | 1.328ms       | 1.269ms       | 0.250ms      | 0.232ms      | 0.019ms  | 0.020ms  |
| 854   | 480    | 4       | 1.717ms       | 1.670ms       | 0.263ms      | 0.262ms      | 0.024ms  | 0.025ms  |
| 1280  | 720    | 1       | 0.873ms       | 0.937ms       | 0.130ms      | 0.180ms      | 0.019ms  | 0.016ms  |
| 1280  | 720    | 3       | 2.877ms       | 2.828ms       | 0.449ms      | 0.457ms      | 0.039ms  | 0.039ms  |
| 1280  | 720    | 4       | 3.558ms       | 3.848ms       | 0.719ms      | 0.616ms      | 0.049ms  | 0.051ms  |
| 1920  | 1080   | 1       | 1.949ms       | 2.136ms       | 0.374ms      | 0.342ms      | 0.036ms  | 0.032ms  |
| 1920  | 1080   | 3       | 6.587ms       | 6.469ms       | 1.000ms      | 0.672ms      | 0.081ms  | 0.084ms  |
| 1920  | 1080   | 4       | 8.144ms       | 8.615ms       | 0.832ms      | 0.914ms      | 0.106ms  | 0.109ms  |
| 2560  | 1440   | 1       | 3.530ms       | 3.800ms       | 0.423ms      | 0.476ms      | 0.061ms  | 0.056ms  |
| 2560  | 1440   | 3       | 11.470ms      | 11.611ms      | 1.323ms      | 1.169ms      | 0.141ms  | 0.144ms  |
| 2560  | 1440   | 4       | 14.139ms      | 15.273ms      | 2.391ms      | 2.567ms      | 0.188ms  | 0.191ms  |
| 3840  | 2160   | 1       | 7.976ms       | 8.494ms       | 1.103ms      | 1.387ms      | 0.134ms  | 0.122ms  |
| 3840  | 2160   | 3       | 26.299ms      | 25.824ms      | 5.339ms      | 4.438ms      | 0.318ms  | 0.329ms  |
| 3840  | 2160   | 4       | 32.941ms      | 34.718ms      | 5.805ms      | 4.514ms      | 0.421ms  | 0.427ms  |
| 7680  | 4320   | 1       | 31.536ms      | 34.100ms      | 5.742ms      | 4.976ms      | 0.527ms  | 0.476ms  |
| 7680  | 4320   | 3       | 102.875ms     | 102.419ms     | 19.261ms     | 17.294ms     | 1.252ms  | 1.290ms  |
| 7680  | 4320   | 4       | 133.081ms     | 136.308ms     | 23.398ms     | 18.445ms     | 1.670ms  | 1.688ms  |

## Contact
For any questions or suggestions, please open an issue or contact the me.
