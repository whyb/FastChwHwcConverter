# FastChwHwcConverter

## Overview
FastChwHwcConverter is a high-performance, header-only C++ library for converting image data formats between HWC (Height, Width, Channels) and CHW (Channels, Height, Width). The library leverages OpenMP parallel processing to provide lightning-fast performance.

## Features
- **High-Performance**: Utilizes OpenMP for parallel processing. Make full use of CPU multi-core features.
- **Header-Only**: Easy to integrate into your C/C++ project.
- **Flexible**: Supports scaling, clamping, and normalization of image data.

## Installation
Simply include the header file `FastChwHwcConverter.hpp` in your project:

```cpp
#include "FastChwHwcConverter.hpp"
```

## Requirements
* C++11 or later
* OpenMP support (optional but recommended for high performance)

## Usage

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

int main() {
    const size_t ch = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output data
    uint8_t src_uint8[h * w * ch]; // Source data(hwc)
    float src_float[h * w * ch]; // Source data(chw)

    float out_float[ch * w * h]; // Inference output data(chw)
    uint8_t out_uint8[ch * w * h]; // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::hwc2chw<uint8_t, float>(ch, w, h, src_uint8, src_float);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::chw2hwc<float, uint8_t>(ch, w, h, out_float, out_uint8);

    return 0;
}
```

## Contact
For any questions or suggestions, please open an issue or contact the maintainer.