# FastChwHwcConverter
![CI](https://github.com/whyb/FastChwHwcConverter/workflows/CI/badge.svg)

## Overview
### Multi-Core CPU Implementation (C++Thread-OpenMP-oneTBB)
FastChwHwcConverter.hpp is a high-performance, multi-threaded, header-only C++ library for converting image data formats between **HWC (Height, Width, Channels)** and **CHW (Channels, Height, Width)**. It leverages `C++ STL Thread` / `OpenMP` / `Intel oneTBB` for parallel processing, utilizing all CPU cores for maximum performance.

**Note**: If the compilation environment does not find OpenMP, or you set USE_OPENMP to OFF, it will be use C++ thread mode.


### GPU Acceleration (NVIDIA CUDA)
FastChwHwcConverterCuda.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, supporting CUDA versions 10.0+ and above. It requires no installation of the CUDA SDK, header files, or static linking. The library dynamically loads CUDA libraries from the system path. It will automatically search for CUDA's dynamic link library from the system path and dynamically load the functions inside and use them.

### GPU Acceleration (AMD ROCm)
FastChwHwcConverterROCm.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, supporting ROCm versions 5.0+ and above. Like the CUDA library, it does not require the ROCm (HIP) SDK, header files, or static linking, and dynamically loads ROCm libraries from the system path.

### GPU Acceleration (Vulkan)
FastChwHwcConverterVulkan.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, using the Vulkan compute API. Like the CUDA and ROCm libraries, it does not require the Vulkan SDK headers or static linking: at runtime it loads `vulkan-1.dll` / `libvulkan.so.1` (or `libMoltenVK.dylib` on macOS) and uses the **glslang** shared library (bundled with the Vulkan Runtime/SDK) to compile the embedded GLSL compute shaders into SPIR-V on first use. It automatically picks the physical device with the strongest compute capability (discrete GPU first) and the most device-local memory.

### GPU Acceleration (OpenCL)
FastChwHwcConverterOpenCL.hpp is a high-performance, GPU-accelerated library for converting image data formats between **HWC** and **CHW**, using the OpenCL compute API. Like the CUDA, ROCm and Vulkan libraries, it does not require the OpenCL SDK headers or static linking: at runtime it loads the system OpenCL ICD loader (`OpenCL.dll` / `libOpenCL.so.1` / `OpenCL.framework` on macOS) and compiles the embedded OpenCL C kernels on first use. It automatically picks the platform/device with the strongest compute capability (GPU first, then more compute units and higher clock frequency). No OpenCL header files are copied into this project: the backend declares its own minimal OpenCL ABI types and dynamically loads everything from the system.

**Note (all GPU backends)**: If your operating environment does not support a GPU backend or does not meet the conditions for using it, it will automatically fall back to the CPU (OpenMP/C++ Thread/Intel oneTBB) for processing. Every GPU backend shares the same template API as the CPU backend (the `HasAlpha` / `NeedClamp` / `NeedNormalizedMeanStds` template parameters and the `alpha` / `min_v` / `max_v` / `mean` / `stds` parameters, any `Stype`/`Dtype`; the GPU accelerates the `uint8_t` and `float` element types and automatically falls back to the CPU for any other type). The functions support passing in host memory or device memory parameters (`CUdeviceptr` for CUDA, `hipDeviceptr_t` for ROCm, `VkBuffer` for Vulkan, `cl_mem` for OpenCL).

Any similar type conversion code you find another project on GitHub will most likely only achieve performance close to the speed of [single-thread execution](#benchmark-performance-timing-results).
## Table of Contents
- [Overview](#overview)
  - [Multi-Core CPU Implementation (C++Thread-OpenMP-oneTBB)](#multi-core-cpu-implementation-cthread-openmp-onetbb)
  - [GPU Acceleration (NVIDIA CUDA)](#gpu-acceleration-nvidia-cuda)
  - [GPU Acceleration (AMD ROCm)](#gpu-acceleration-amd-rocm)
  - [GPU Acceleration (Vulkan)](#gpu-acceleration-vulkan)
  - [GPU Acceleration (OpenCL)](#gpu-acceleration-opencl)
- [The difference between CHW and HWC](#the-difference-between-chw-and-hwc)
  - [CHW Format](#chw-format)
  - [HWC Format](#hwc-format)
- [Why Convert Between HWC and CHW Formats?](#why-convert-between-hwc-and-chw-formats)
- [Features](#features)
- [Installation](#installation)
  - [for CPU (C++ Thread)](#for-cpu-c-thread)
  - [for CPU (OpenMP)](#for-cpu-openmp)
  - [for CPU (oneTBB)](#for-cpu-onetbb)
  - [for GPU (CUDA / ROCm / Vulkan / OpenCL)](#for-gpu-cuda--rocm--vulkan--opencl)
- [Requirements](#requirements)
- [API Documents](#api-documents)
  - [HWC -> CHW Conversion](#hwc-to-chw-conversion)
  - [CHW -> HWC Conversion](#chw-to-hwc-conversion)
  - [Example](#example)
- [Validation](#validation)
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
- **High-Performance**: Utilizes C++ Thread / OpenMP / Intel oneTBB for parallel processing. Make full use of CPU multi-core features.
- **GPU Optimization**: Fully leverages NVIDIA CUDA, AMD ROCm, Vulkan and OpenCL to harness the computational power of GPUs, accelerating performance for intensive workloads.
- **Header-Only**: Include **ONLY** a single header file. Easy to integrate into your C/C++ project. [example](#example).
- **Flexible**: Supports scaling, clamping, and normalization of image data, any data type. The CPU and all GPU backends share one template API (`HasAlpha` / `NeedClamp` / `NeedNormalizedMeanStds`).
- **Lightweight & SDK-Free**: No dependency on any external SDKs like CUDA SDK or HIP SDK. The project requires no additional header files or static library linkage, making it clean and easy to deploy.

## Installation
### for CPU (C++ Thread)
Simply include the header file `FastChwHwcConverter.hpp` in your project:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DUSE_TBB=OFF -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=OFF -DBUILD_ROCM_BENCHMARK=OFF -DBUILD_EXAMPLE=OFF -DBUILD_EXAMPLE_OPENCV=OFF

cmake --build build --config Release
```

### for CPU (OpenMP)
OpenMP is an API that supports multi-platform shared-memory multiprocessing programming. on many platforms, instruction-set architectures and operating systems. OpenMP uses a portable, scalable model that gives programmers a simple and flexible interface for developing parallel applications for platforms ranging from the standard desktop computer to the supercomputer.
[see more](https://www.openmp.org).

 * Option 1:

    ```shell
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DUSE_TBB=OFF -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=OFF -DBUILD_ROCM_BENCHMARK=OFF -DBUILD_EXAMPLE=OFF -DBUILD_EXAMPLE_OPENCV=OFF

    cmake --build build --config Release
    ```
 * Option 2:

    Simply include the header file `FastChwHwcConverter.hpp` in your project. Before include, you need to add a macro `#define USE_OPENMP 1`.

### for CPU (oneTBB)
Intel oneTBB (Intel® oneAPI Threading Building Blocks) is a simplify parallelism with this advanced threading and memory-management template library. This component is part of the Intel® oneAPI Base Toolkit. [see more](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb-download.html).

 * Option 1:

    ```shell
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DUSE_TBB=ON -DTBB_DIR=D:/extlibs/oneAPI/tbb/2021.13/lib/cmake/tbb -DBUILD_BENCHMARK=ON -DBUILD_CUDA_BENCHMARK=OFF -DBUILD_ROCM_BENCHMARK=OFF -DBUILD_EXAMPLE=OFF -DBUILD_EXAMPLE_OPENCV=OFF

    cmake --build build --config Release
    ```
 * Option 2:

    Simply include the header file `FastChwHwcConverter.hpp` in your project. Before include, you need to add a macro `#define USE_TBB 1`.


### for GPU (CUDA / ROCm / Vulkan / OpenCL)

All GPU backends are SDK-free at build time, the CMake switches only control which benchmark targets are generated:

 * Option 1:

    ```shell
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_BENCHMARK=ON -DBUILD_ROCM_BENCHMARK=ON -DBUILD_VULKAN_BENCHMARK=ON -DBUILD_OPENCL_BENCHMARK=ON

    cmake --build build --config Release
    ```

    The Vulkan benchmark target is only generated when the Vulkan SDK is found by `find_package(Vulkan)` (set `Vulkan_INCLUDE_DIR` / `Vulkan_LIBRARY` or the `VULKAN_SDK` environment variable if it is not detected automatically). If the SDK is not found, the Vulkan target is skipped.

 * Option 2:

    Simply include the header file of the backend you need in your project:

    ```cpp
    #include "FastChwHwcConverterCuda.hpp"
    #include "FastChwHwcConverterROCm.hpp"
    #include "FastChwHwcConverterVulkan.hpp"
    #include "FastChwHwcConverterOpenCL.hpp"
    ```

Runtime requirements per backend:

 * **CUDA** ([NVIDIA CUDA Official Website](https://developer.nvidia.com/cuda-toolkit)): usually you also need to copy the `nvrtc64_***_0.dll` `nvrtc-builtins64_***` (for Windows) or `libnvrtc.so` (for Linux) file in the CUDA Runtime SDK to the executable program directory, or set the CUDA SDK HOME as a system environment variable. In addition, you need to download and install the latest version of the driver from the [NVIDIA drivers website](https://www.nvidia.com/Download/index.aspx), because this project will dynamically load the driver file: `nvcuda.dll` (for Windows) or `libcuda.so` (for Linux).
 * **ROCm** ([AMD ROCm Official Website](https://www.amd.com/en/products/software/rocm.html)): usually you also need to copy the `hiprtc****.dll` `hiprtc-builtins****.dll` `amd_comgr_*.dll` (for Windows) or `libhiprtc.so` (for Linux) file in the ROCm Runtime SDK to the executable program directory, or set the ROCm SDK HOME as a system environment variable. In addition, you need to download and install the latest version of the driver from the [AMD drivers website](https://www.amd.com/en/support), because this project will dynamically load the driver file: `amdhip64_6.dll` (for Windows) or `libamdhip64.so` (for Linux).
 * **Vulkan** ([Vulkan SDK Official Website](https://vulkan.lunarg.com/sdk/home)): at runtime the backend loads `vulkan-1.dll` / `libvulkan.so.1` (or `libMoltenVK.dylib` on macOS) and the `glslang` shared library (searched next to the executable, in `$VULKAN_SDK/Bin` on Windows or `$VULKAN_SDK/lib` on Linux/macOS, then in the system paths) to compile the GLSL compute shaders into SPIR-V.
 * **OpenCL**: no SDK or header files are required at build time (the `BUILD_OPENCL_BENCHMARK` switch only controls whether the target is generated); at runtime the backend loads the system OpenCL ICD loader (`OpenCL.dll` / `libOpenCL.so.1` / `OpenCL.framework` on macOS) and compiles the embedded OpenCL C kernels for the selected device.
## Requirements
* C++17 or later
* OpenMP support (optional, set USE_OPENMP to ON for high performance)
* oneTBB support (optional, set USE_TBB to ON and set valid TBB_LIBS for Intel oneTBB's high performance)
* CMake v3.10 or later (optional)
* OpenCV v4.0 or later (optional, if BUILD_EXAMPLE_OPENCV is ON)
* CUDA 11.2+ driver (optional, if you want to use CUDA acceleration, And NVIDIA GPU's compute capability > 3.5, more details see [here](https://developer.nvidia.com/cuda-gpus). )
* ROCm 5.0+ driver (optional, if you want to use ROCm acceleration, hardware and system requirements see [here](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html). )
* Vulkan 1.0 support (optional, if you want to use Vulkan acceleration, see [here](https://vulkan.lunarg.com/sdk/home). )
* OpenCL 1.2+ support (optional, if you want to use OpenCL acceleration, only an OpenCL runtime/ICD loader is needed at runtime, no OpenCL SDK or header files are required. )

## API Documents

All backends (CPU and GPU) share one template API. The GPU backends `whyb::nvidia::` (CUDA) / `whyb::amd::` (ROCm) / `whyb::vulkan::` / `whyb::opencl::` expose the same templates as `whyb::cpu::`, plus device-memory overloads, and compute the exact same transform:

    value = HasAlpha ? src * alpha : src
    value = NeedNormalizedMeanStds ? (value - mean[min(channel, 2)]) / stds[min(channel, 2)] : value
    value = NeedClamp ? clamp(value, min_v, max_v) : value
    dst = (Dtype)value

The GPU kernels cover the `uint8_t` and `float` element types; any other `Stype`/`Dtype` combination keeps full CPU behavior through the automatic fallback.

### HWC to CHW Conversion
The `hwc2chw()` function converts image data from HWC format to CHW format.

Host memory overload (`whyb::cpu::` / `whyb::nvidia::` / `whyb::amd::` / `whyb::vulkan::` / `whyb::opencl::`):
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

Device memory overload (GPU backends only, `Stype`/`Dtype` must be given explicitly):
```cpp
template <typename Stype, typename Dtype,
            bool HasAlpha = false,
            bool NeedClamp = false,
            bool NeedNormalizedMeanStds = false>
void hwc2chw(
    const size_t h, const size_t w, const size_t c,
    const CUdeviceptr src, const CUdeviceptr dst,
    const Dtype alpha = 1,
    const Dtype min_v = 0.0, const Dtype max_v = 1.0,
    const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
    const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }
);
```

The device handle type depends on the backend: `CUdeviceptr` for `whyb::nvidia::`, `hipDeviceptr_t` for `whyb::amd::`, `VkBuffer` for `whyb::vulkan::`, `cl_mem` for `whyb::opencl::`.

Template parameters:

* `Stype`: Source data type (any type on the CPU, `uint8_t` / `float` are GPU-accelerated).
* `Dtype`: Destination data type (any type on the CPU, `uint8_t` / `float` are GPU-accelerated).
* `HasAlpha`: input alpha is requires participation in calculations.
* `NeedClamp`: input min_v and max_v are requires participation in calculations.
* `NeedNormalizedMeanStds`: input mean and stds are requires participation in calculations.

Parameters:

* `h`: Height of the image.
* `w`: Width of the image.
* `c`: Number of channels.
* `src`: Source data in HWC format (host memory or device memory).
* `dst`: Destination data in CHW format (host memory or device memory).
* `alpha`: Scaling factor (default is 1).
* `min_v`: Minimum value for clamping (default is 0.0).
* `max_v`: Maximum value for clamping (default is 1.0).
* `mean`: Array of mean values for normalization (default is {0.485, 0.456, 0.406}).
* `stds`: Array of standard deviation values for normalization (default is {0.229, 0.224, 0.225}).

Normalization parameters are RGB-only. A fourth channel reuses the final parameter set to avoid reading outside the arrays.

### CHW to HWC Conversion
The `chw2hwc()` function converts image data from CHW format to HWC format.

Host memory overload (`whyb::cpu::` / `whyb::nvidia::` / `whyb::amd::` / `whyb::vulkan::` / `whyb::opencl::`):
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

Device memory overload (GPU backends only, `Stype`/`Dtype` must be given explicitly):
```cpp
template <typename Stype, typename Dtype,
            bool HasAlpha = false,
            bool NeedClamp = false>
void chw2hwc(
    const size_t c, const size_t h, const size_t w,
    const CUdeviceptr src, const CUdeviceptr dst,
    const Dtype alpha = 1,
    const Dtype min_v = 0, const Dtype max_v = 255
);
```

The device handle type depends on the backend: `CUdeviceptr` for `whyb::nvidia::`, `hipDeviceptr_t` for `whyb::amd::`, `VkBuffer` for `whyb::vulkan::`, `cl_mem` for `whyb::opencl::`.

Template parameters:

* `Stype`: Source data type (any type on the CPU, `uint8_t` / `float` are GPU-accelerated).
* `Dtype`: Destination data type (any type on the CPU, `uint8_t` / `float` are GPU-accelerated).
* `HasAlpha`: input alpha is requires participation in calculations.
* `NeedClamp`: input min_v and max_v are requires participation in calculations.

Parameters:

* `c`: Number of channels.
* `h`: Height of the image.
* `w`: Width of the image.
* `src`: Source data in CHW format (host memory or device memory).
* `dst`: Destination data in HWC format (host memory or device memory).
* `alpha`: Scaling factor (default is 1).
* `min_v`: Minimum value for clamping (default is 0).
* `max_v`: Maximum value for clamping (default is 255).

**Note (GPU backends)**: Please call `whyb::nvidia::init()` / `whyb::amd::init()` / `whyb::vulkan::init()` / `whyb::opencl::init()` before the first use, and call `release()` to release it after confirming that it will not be used anymore. The host memory overload falls back to the CPU implementation automatically if the GPU backend is not available. Device buffers can be allocated with `createDeviceBuffer()` and must be released with `destroyDeviceBuffer()` (the Vulkan and OpenCL backends also provide `uploadToDeviceBuffer()` / `downloadFromDeviceBuffer()`).

### Example
This example code(**test/example.cpp**) demonstrates how to use the CPU, NVIDIA CUDA, AMD ROCm, Vulkan and OpenCL backends to convert image data from HWC format to CHW format, and then back to HWC format after AI inference. All backends use the same template API, only the class name differs.

```cpp
#include "FastChwHwcConverter.hpp"
#include "FastChwHwcConverterCuda.hpp"
#include "FastChwHwcConverterOpenCL.hpp"
#include "FastChwHwcConverterROCm.hpp"
#include "FastChwHwcConverterVulkan.hpp"
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
    whyb::cpu::hwc2chw<uint8_t, float, true>(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::cpu::chw2hwc<float, uint8_t, true>(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

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
    whyb::nvidia::hwc2chw<uint8_t, float, true>(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::nvidia::chw2hwc<float, uint8_t, true>(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    whyb::nvidia::release();
    std::cout << "cuda example done" << std::endl;
}
```

The `rocm_example()` / `vulkan_example()` / `opencl_example()` functions are identical to `cuda_example()` except for the class name (`whyb::amd::` / `whyb::vulkan::` / `whyb::opencl::`), see the full code in **test/example.cpp**.

If you are using OpenCV's `cv::Mat`, Please refer to the **test/example-opencv.cpp** file.
## Validation

test/validate.cpp checks correctness with randomized inputs (no third-party test framework is used):

* The CPU backend is compared element-by-element against a naive scalar reference implementation for every HasAlpha / NeedClamp / NeedNormalizedMeanStds template branch, with channels 1, 3 and 4 (normalization is only defined for up to 3 channels).
* The CUDA / ROCm / Vulkan / OpenCL backends, when a device is available, are compared against the CPU backend on the same random data. The comparison sweeps the full `Stype` x `Dtype` x `HasAlpha` x `NeedClamp` (x `NeedNormalizedMeanStds`) permutation matrix over the basic data types (`bool`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, `double`), plus non-default `alpha` / `min_v` / `max_v` / `mean` / `stds` values. The GPU kernels accelerate the `uint8_t`/`float` element types and route every other combination through the automatic CPU fallback, so the matrix validates both the GPU math and the type dispatch.

Build and run it with (the target is enabled by default):

```shell
cmake -S . -B build -DBUILD_VALIDATE=ON

cmake --build build --config Release

build/validate
```

On Windows the binary is `build\Release\validate.exe`. The program prints a summary for every backend and exits with a non-zero code if any mismatch is found. A backend without a GPU/device is skipped gracefully (exit code 0), so the CPU checks still run on CI machines that have no GPU.

## Benchmark Performance Timing Results

The table below shows the benchmark performance timing for different image dimensions, channels, and processing configurations. The GPU columns are measured on device memory, while the CPU columns are measured on host memory.

All benchmark programs fill their inputs with randomized data, run warm-up iterations, and report the minimum average over several repetitions (see `test/benchmark.cpp` and `test/*_benchmark.cpp`), so the results are not biased by all-zero buffers or one-off system noise.

    RAM: DDR5 4800MHz 4x16GB
    CPU(OpenMP): Intel(R) Core(TM) i7-13700K
    GPU(CUDA): NVIDIA GeForce RTX 3060 Ti
    GPU(ROCm): AMD Radeon RX 7900 XTX
    GPU(Vulkan): AMD Radeon RX 7900 XTX

|             |CPU(Single)|CPU(Single)|CPU(OpenMP)|CPU(OpenMP)|   CUDA  |   CUDA  |   ROCm  |   ROCm  |  Vulkan |  Vulkan |
|-------------|----------|----------|----------|----------|---------|---------|---------|---------|---------|---------|
|  W x H x C  |  hwc2chw |  chw2hwc |  hwc2chw |  chw2hwc |  hwc2chw|  chw2hwc|  hwc2chw|  chw2hwc| hwc2chw | chw2hwc |
| 426x240x1 | 0.100ms | 0.117ms | 0.127ms | 0.029ms | 0.017ms | 0.018ms | 0.179ms | 0.177ms | 0.051ms | 0.049ms |
| 426x240x3 | 0.359ms | 0.402ms | 0.067ms | 0.068ms | 0.019ms | 0.019ms | 0.179ms | 0.179ms | 0.046ms | 0.046ms |
| 426x240x4 | 0.439ms | 0.539ms | 0.076ms | 0.080ms | 0.019ms | 0.019ms | 0.181ms | 0.187ms | 0.048ms | 0.048ms |
| 640x360x1 | 0.227ms | 0.262ms | 0.050ms | 0.051ms | 0.020ms | 0.020ms | 0.185ms | 0.184ms | 0.048ms | 0.048ms |
| 640x360x3 | 0.753ms | 0.878ms | 0.083ms | 0.072ms | 0.024ms | 0.027ms | 0.200ms | 0.198ms | 0.044ms | 0.042ms |
| 640x360x4 | 0.973ms | 1.227ms | 0.085ms | 0.093ms | 0.025ms | 0.025ms | 0.205ms | 0.201ms | 0.059ms | 0.043ms |
| 854x480x1 | 0.402ms | 0.470ms | 0.048ms | 0.047ms | 0.022ms | 0.022ms | 0.202ms | 0.201ms | 0.044ms | 0.048ms |
| 854x480x3 | 1.326ms | 1.562ms | 0.136ms | 0.135ms | 0.031ms | 0.038ms | 0.207ms | 0.166ms | 0.062ms | 0.046ms |
| 854x480x4 | 1.733ms | 2.170ms | 0.236ms | 0.277ms | 0.035ms | 0.036ms | 0.133ms | 0.131ms | 0.045ms | 0.047ms |
| 1280x720x1 | 0.898ms | 1.048ms | 0.174ms | 0.127ms | 0.039ms | 0.040ms | 0.131ms | 0.133ms | 0.044ms | 0.047ms |
| 1280x720x3 | 3.058ms | 3.506ms | 0.288ms | 0.285ms | 0.054ms | 0.068ms | 0.138ms | 0.136ms | 0.059ms | 0.061ms |
| 1280x720x4 | 3.934ms | 4.988ms | 0.388ms | 0.381ms | 0.063ms | 0.060ms | 0.139ms | 0.138ms | 0.058ms | 0.058ms |
| 1920x1080x1 | 2.052ms | 2.392ms | 0.247ms | 0.221ms | 0.068ms | 0.070ms | 0.146ms | 0.143ms | 0.058ms | 0.057ms |
| 1920x1080x3 | 6.763ms | 7.951ms | 0.718ms | 0.701ms | 0.102ms | 0.129ms | 0.149ms | 0.115ms | 0.058ms | 0.067ms |
| 1920x1080x4 | 8.738ms | 11.238ms | 1.068ms | 0.968ms | 0.124ms | 0.117ms | 0.116ms | 0.111ms | 0.064ms | 0.074ms |
| 2560x1440x1 | 3.674ms | 4.238ms | 0.505ms | 0.394ms | 0.108ms | 0.114ms | 0.117ms | 0.118ms | 0.059ms | 0.061ms |
| 2560x1440x3 | 11.905ms | 14.097ms | 1.233ms | 1.356ms | 0.170ms | 0.207ms | 0.115ms | 0.113ms | 0.072ms | 0.084ms |
| 2560x1440x4 | 15.574ms | 20.065ms | 2.055ms | 2.093ms | 0.213ms | 0.199ms | 0.113ms | 0.114ms | 0.079ms | 0.097ms |
| 3840x2160x1 | 8.295ms | 9.449ms | 1.491ms | 1.476ms | 0.235ms | 0.235ms | 0.113ms | 0.113ms | 0.069ms | 0.074ms |
| 3840x2160x3 | 26.690ms | 31.392ms | 3.907ms | 3.340ms | 0.367ms | 0.467ms | 0.298ms | 0.238ms | 0.231ms | 0.245ms |
| 3840x2160x4 | 35.655ms | 46.030ms | 5.580ms | 4.854ms | 0.449ms | 0.425ms | 0.393ms | 0.306ms | 0.298ms | 0.303ms |
| 7680x4320x1 | 33.288ms | 38.398ms | 5.516ms | 3.910ms | 0.885ms | 0.878ms | 0.324ms | 0.310ms | 0.269ms | 0.305ms |
| 7680x4320x3 | 107.704ms | 125.577ms | 17.621ms | 13.476ms | 1.410ms | 1.792ms | 1.013ms | 0.755ms | 0.781ms | 0.848ms |
| 7680x4320x4 | 140.619ms | 182.024ms | 21.929ms | 17.257ms | 1.713ms | 1.618ms | 1.475ms | 0.990ms | 1.040ms | 1.083ms |

## Contact
For any questions or suggestions, please feel free to open an issue or reach out to me.
