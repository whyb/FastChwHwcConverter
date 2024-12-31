# FastChwHwcConverter

## Overview
FastChwHwcConverter is a high-performance, multi-thread, header-only C++ library for converting image data formats between **HWC (Height, Width, Channels)** and **CHW (Channels, Height, Width)**. The library leverages OpenMP parallel processing to provide lightning-fast performance.

Any similar type conversion code you find another project on GitHub will most likely only achieve performance close to the speed of [single-thread execution](#benchmark-performance-timing-results).

## Features
- **High-Performance**: Utilizes OpenMP for parallel processing. Make full use of CPU multi-core features.
- **Header-Only**: Include **ONLY** a single header file. Easy to integrate into your C/C++ project. [example](#example).
- **Flexible**: Supports scaling, clamping, and normalization of image data, any data type.

## Installation
Simply include the header file `FastChwHwcConverter.hpp` in your project:

```cpp
#include "FastChwHwcConverter.hpp"
```

## Requirements
* C++11 or later
* OpenMP support (optional but recommended for high performance)

## Let's Converter

### HWC to CHW Conversion
The `hwc2chw` function converts image data from HWC format to CHW format.
```cpp
template <typename Stype, typename Dtype>
void hwc2chw(
    const size_t ch, const size_t w, const size_t h,
    const Stype* src, Dtype* dst,
    const Dtype alpha = 1, const bool clamp = false,
    const Dtype min_v = 0.0, const Dtype max_v = 1.0,
    const bool normalized_mean_stds = false,
    const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
    const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }
);
```

Parameters:

* `ch`: Number of channels.
* `w`: Width of the image.
* `h`: Height of the image.
* `src`: Pointer to the source data in HWC format.
* `dst`: Pointer to the destination data in CHW format.
* `alpha`: Scaling factor (default is 1).
* `clamp`: Whether to clamp the data values (default is false).
* `min_v`: Minimum value for clamping (default is 0.0).
* `max_v`: Maximum value for clamping (default is 1.0).
* `normalized_mean_stds`: Whether to use mean and standard deviation for normalization (default is false).
* `mean`: Array of mean values for normalization (default is {0.485, 0.456, 0.406}).
* `stds`: Array of standard deviation values for normalization (default is {0.229, 0.224, 0.225}).

### CHW to HWC Conversion
The `chw2hwc` function converts image data from CHW format to HWC format.

```cpp
template <typename Stype, typename Dtype>
void chw2hwc(
    const size_t ch, const size_t w, const size_t h, 
    const Stype* src, Dtype* dst, 
    const double alpha = 1, const bool clamp = false,
    const Dtype min_v = 0, const Dtype max_v = 255
);
```
Parameters:

* `ch`: Number of channels.
* `w`: Width of the image.
* `h`: Height of the image.
* `src`: Pointer to the source data in CHW format.
* `dst`: Pointer to the destination data in HWC format.
* `alpha`: Scaling factor (default is 1).
* `clamp`: Whether to clamp the data values (default is false).
* `min_v`: Minimum value for clamping (default is 0).
* `max_v`: Maximum value for clamping (default is 255).


### Example
```cpp
#include "FastChwHwcConverter.hpp"
#include <vector>

int main() {
    const size_t ch = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * ch;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::hwc2chw<uint8_t, float>(ch, w, h, (uint8_t*)src_uint8.data(), (float*)src_float.data());

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::chw2hwc<float, uint8_t>(ch, w, h, (float*)out_float.data(), (uint8_t*)out_uint8.data());

    return 0;
}
```

## Benchmark Performance Timing Results

The table below shows the benchmark performance timing for different image dimensions, channels, and processing configurations.

CPU: Intel(R) Core(TM) i7-13700K
RAM: DDR5 2400MHz 4x32-bit channels

|       |        |         | single-thread | single-thread | multi-thread | multi-thread |
|-------|--------|---------|---------------|---------------|--------------|--------------|
| Width | Height | Channel | hwc2chw       | chw2hwc       | hwc2chw      | chw2hwc      |
| 426   | 240    | 1       | 0ms           | 0ms           | 0ms          | 0ms          |
| 426   | 240    | 3       | 0ms           | 0ms           | 0ms          | 0ms          |
| 426   | 240    | 4       | 0ms           | 0ms           | 0ms          | 0ms          |
| 640   | 360    | 1       | 0ms           | 0ms           | 0ms          | 0ms          |
| 640   | 360    | 3       | 0ms           | 0ms           | 0ms          | 0ms          |
| 640   | 360    | 4       | 0ms           | 0ms           | 0ms          | 0ms          |
| 854   | 480    | 1       | 0ms           | 0ms           | 0ms          | 0ms          |
| 854   | 480    | 3       | 1ms           | 1ms           | 0ms          | 0ms          |
| 854   | 480    | 4       | 1ms           | 1ms           | 0ms          | 0ms          |
| 1280  | 720    | 1       | 0ms           | 1ms           | 0ms          | 0ms          |
| 1280  | 720    | 3       | 2ms           | 2ms           | 0ms          | 0ms          |
| 1280  | 720    | 4       | 3ms           | 3ms           | 0ms          | 0ms          |
| 1920  | 1080   | 1       | 1ms           | 2ms           | 0ms          | 0ms          |
| 1920  | 1080   | 3       | 6ms           | 6ms           | 0ms          | 0ms          |
| 1920  | 1080   | 4       | 8ms           | 8ms           | 1ms          | 1ms          |
| 2560  | 1440   | 1       | 3ms           | 3ms           | 0ms          | 0ms          |
| 2560  | 1440   | 3       | 11ms          | 11ms          | 1ms          | 1ms          |
| 2560  | 1440   | 4       | 14ms          | 14ms          | 2ms          | 1ms          |
| 3840  | 2160   | 1       | 7ms           | 8ms           | 0ms          | 1ms          |
| 3840  | 2160   | 3       | 25ms          | 25ms          | 5ms          | 3ms          |
| 3840  | 2160   | 4       | 32ms          | 32ms          | 6ms          | 4ms          |
| 7680  | 4320   | 1       | 31ms          | 35ms          | 5ms          | 4ms          |
| 7680  | 4320   | 3       | 102ms         | 101ms         | 19ms         | 13ms         |
| 7680  | 4320   | 4       | 129ms         | 126ms         | 25ms         | 20ms         |

## Contact
For any questions or suggestions, please open an issue or contact the maintainer.
