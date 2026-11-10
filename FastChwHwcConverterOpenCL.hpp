/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2025-2026] [張小凡](https://github.com/whyb)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#pragma once

#include "DynamicLibraryManager.hpp"
#include "FastChwHwcConverter.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <mutex>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif


namespace whyb {

// OpenCL API calling convention define (cl_platform.h counterpart)
#ifndef CL_API_CALL
#ifdef _WIN32
#define CL_API_CALL __stdcall
#else
#define CL_API_CALL
#endif
#endif
#ifndef CL_CALLBACK
#ifdef _WIN32
#define CL_CALLBACK __stdcall
#else
#define CL_CALLBACK
#endif
#endif

// ---------------------------------------------------------------------------
// Minimal OpenCL ABI types (cl.h / cl_platform.h counterparts).
// These declarations must stay byte-for-byte compatible with the OpenCL ICD
// loader that is loaded at runtime. The OpenCL library is loaded dynamically
// (like NVRTC/HIPRTC in the CUDA/ROCm backends), so no OpenCL headers or
// link-time dependency are required.
// ---------------------------------------------------------------------------

// OpenCL scalar data types
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef int64_t cl_long;
typedef uint64_t cl_ulong;
typedef uint16_t cl_half;
typedef float cl_float;
typedef double cl_double;
typedef intptr_t cl_context_properties;

// OpenCL opaque handle types
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;
typedef struct _cl_sampler* cl_sampler;

// OpenCL enum / bitfield types
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_program_build_info;

// OpenCL error codes (frozen OpenCL ABI constants)
#ifndef CL_SUCCESS
#define CL_SUCCESS 0

#endif
#ifndef CL_DEVICE_NOT_FOUND
#define CL_DEVICE_NOT_FOUND -1

#endif

// cl_bool values
#ifndef CL_FALSE
#define CL_FALSE 0

#endif
#ifndef CL_TRUE
#define CL_TRUE 1

#endif

// cl_device_type bits
#ifndef CL_DEVICE_TYPE_DEFAULT
#define CL_DEVICE_TYPE_DEFAULT (1 << 0)

#endif
#ifndef CL_DEVICE_TYPE_CPU
#define CL_DEVICE_TYPE_CPU (1 << 1)

#endif
#ifndef CL_DEVICE_TYPE_GPU
#define CL_DEVICE_TYPE_GPU (1 << 2)

#endif
#ifndef CL_DEVICE_TYPE_ACCELERATOR
#define CL_DEVICE_TYPE_ACCELERATOR (1 << 3)

#endif
#ifndef CL_DEVICE_TYPE_CUSTOM
#define CL_DEVICE_TYPE_CUSTOM (1 << 4)

#endif
#ifndef CL_DEVICE_TYPE_ALL
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF

#endif

// cl_mem_flags bits
#ifndef CL_MEM_READ_WRITE
#define CL_MEM_READ_WRITE (1 << 0)

#endif
#ifndef CL_MEM_WRITE_ONLY
#define CL_MEM_WRITE_ONLY (1 << 1)

#endif
#ifndef CL_MEM_READ_ONLY
#define CL_MEM_READ_ONLY (1 << 2)

#endif
#ifndef CL_MEM_USE_HOST_PTR
#define CL_MEM_USE_HOST_PTR (1 << 3)

#endif
#ifndef CL_MEM_ALLOC_HOST_PTR
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)

#endif
#ifndef CL_MEM_COPY_HOST_PTR
#define CL_MEM_COPY_HOST_PTR (1 << 5)

#endif

// cl_command_queue_properties bits
#ifndef CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (1 << 0)

#endif
#ifndef CL_QUEUE_PROFILING_ENABLE
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)

#endif

// cl_platform_info values
#ifndef CL_PLATFORM_PROFILE
#define CL_PLATFORM_PROFILE 0x0900

#endif
#ifndef CL_PLATFORM_VERSION
#define CL_PLATFORM_VERSION 0x0901

#endif
#ifndef CL_PLATFORM_NAME
#define CL_PLATFORM_NAME 0x0902

#endif
#ifndef CL_PLATFORM_VENDOR
#define CL_PLATFORM_VENDOR 0x0903

#endif
#ifndef CL_PLATFORM_EXTENSIONS
#define CL_PLATFORM_EXTENSIONS 0x0904

#endif
#ifndef CL_PLATFORM_NUMERIC_VERSION
#define CL_PLATFORM_NUMERIC_VERSION 0x0906

#endif

// cl_device_info values
#ifndef CL_DEVICE_TYPE
#define CL_DEVICE_TYPE 0x1000

#endif
#ifndef CL_DEVICE_MAX_COMPUTE_UNITS
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002

#endif
#ifndef CL_DEVICE_MAX_WORK_GROUP_SIZE
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004

#endif
#ifndef CL_DEVICE_MAX_WORK_ITEM_SIZES
#define CL_DEVICE_MAX_WORK_ITEM_SIZES 0x1005

#endif
#ifndef CL_DEVICE_MAX_CLOCK_FREQUENCY
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C

#endif
#ifndef CL_DEVICE_ADDRESS_BITS
#define CL_DEVICE_ADDRESS_BITS 0x100D

#endif
#ifndef CL_DEVICE_MAX_MEM_ALLOC_SIZE
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010

#endif
#ifndef CL_DEVICE_GLOBAL_MEM_SIZE
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F

#endif
#ifndef CL_DEVICE_LOCAL_MEM_SIZE
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023

#endif
#ifndef CL_DEVICE_NAME
#define CL_DEVICE_NAME 0x102B

#endif
#ifndef CL_DEVICE_VENDOR
#define CL_DEVICE_VENDOR 0x102C

#endif
#ifndef CL_DEVICE_PROFILE
#define CL_DEVICE_PROFILE 0x102E

#endif
#ifndef CL_DEVICE_VERSION
#define CL_DEVICE_VERSION 0x102F

#endif
#ifndef CL_DEVICE_EXTENSIONS
#define CL_DEVICE_EXTENSIONS 0x1030

#endif
#ifndef CL_DEVICE_PLATFORM
#define CL_DEVICE_PLATFORM 0x1031

#endif
#ifndef CL_DEVICE_OPENCL_C_VERSION
#define CL_DEVICE_OPENCL_C_VERSION 0x103D

#endif

// cl_context_properties values
#ifndef CL_CONTEXT_PLATFORM
#define CL_CONTEXT_PLATFORM 0x1084

#endif

// cl_program_build_info values
#ifndef CL_PROGRAM_BUILD_STATUS
#define CL_PROGRAM_BUILD_STATUS 0x1181

#endif
#ifndef CL_PROGRAM_BUILD_OPTIONS
#define CL_PROGRAM_BUILD_OPTIONS 0x1182

#endif
#ifndef CL_PROGRAM_BUILD_LOG
#define CL_PROGRAM_BUILD_LOG 0x1183

#endif

// OpenCL callback types
typedef void (CL_CALLBACK* clContextNotifyCallback_t)(const char* errinfo, const void* private_info, size_t cb, void* user_data);
typedef void (CL_CALLBACK* clProgramNotifyCallback_t)(cl_program program, void* user_data);

// OpenCL API function type define
typedef cl_int (CL_API_CALL* clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL* clGetPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL* clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL* clGetDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (CL_API_CALL* clCreateContext_t)(const cl_context_properties*, cl_uint, const cl_device_id*, clContextNotifyCallback_t, void*, cl_int*);
typedef cl_int (CL_API_CALL* clReleaseContext_t)(cl_context);
typedef cl_command_queue (CL_API_CALL* clCreateCommandQueue_t)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_int (CL_API_CALL* clReleaseCommandQueue_t)(cl_command_queue);
typedef cl_mem (CL_API_CALL* clCreateBuffer_t)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (CL_API_CALL* clReleaseMemObject_t)(cl_mem);
typedef cl_program (CL_API_CALL* clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL* clBuildProgram_t)(cl_program, cl_uint, const cl_device_id*, const char*, clProgramNotifyCallback_t, void*);
typedef cl_int (CL_API_CALL* clGetProgramBuildInfo_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL* clReleaseProgram_t)(cl_program);
typedef cl_kernel (CL_API_CALL* clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_int (CL_API_CALL* clReleaseKernel_t)(cl_kernel);
typedef cl_int (CL_API_CALL* clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL* clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL* clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL* clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL* clFinish_t)(cl_command_queue);

// OpenCL API function pointers. These are inline variables (one shared
// object across translation units) instead of file-scope statics, so every TU
// that includes this header observes the same initialized set.
inline clGetPlatformIDs_t clGetPlatformIDs = nullptr;
inline clGetPlatformInfo_t clGetPlatformInfo = nullptr;
inline clGetDeviceIDs_t clGetDeviceIDs = nullptr;
inline clGetDeviceInfo_t clGetDeviceInfo = nullptr;
inline clCreateContext_t clCreateContext = nullptr;
inline clReleaseContext_t clReleaseContext = nullptr;
inline clCreateCommandQueue_t clCreateCommandQueue = nullptr;
inline clReleaseCommandQueue_t clReleaseCommandQueue = nullptr;
inline clCreateBuffer_t clCreateBuffer = nullptr;
inline clReleaseMemObject_t clReleaseMemObject = nullptr;
inline clCreateProgramWithSource_t clCreateProgramWithSource = nullptr;
inline clBuildProgram_t clBuildProgram = nullptr;
inline clGetProgramBuildInfo_t clGetProgramBuildInfo = nullptr;
inline clReleaseProgram_t clReleaseProgram = nullptr;
inline clCreateKernel_t clCreateKernel = nullptr;
inline clReleaseKernel_t clReleaseKernel = nullptr;
inline clSetKernelArg_t clSetKernelArg = nullptr;
inline clEnqueueNDRangeKernel_t clEnqueueNDRangeKernel = nullptr;
inline clEnqueueReadBuffer_t clEnqueueReadBuffer = nullptr;
inline clEnqueueWriteBuffer_t clEnqueueWriteBuffer = nullptr;
inline clFinish_t clFinish = nullptr;

// ---------------------------------------------------------------------------
// OpenCL C kernel sources (compiled to device binaries at runtime with
// clBuildProgram).
//
// The kernels mirror the CPU implementation exactly:
//   value = has_alpha ? src * alpha : src
//   value = need_norm ? (value - mean[min(ch,2)]) / stds[min(ch,2)] : value
//   value = need_clamp ? clamp(value, min_v, max_v) : value
//   dst = (Dtype)value
// src_is_float / dst_is_float select the uint8_t or float element type of the
// buffers, so all uint8_t/float source/destination combinations are covered.
// ---------------------------------------------------------------------------
static const char* openclSource = R"(
  // HWC -> CHW
  __kernel void opencl_hwc2chw(const ulong h, const ulong w, const ulong c,
                               __global const uchar* src, __global uchar* dst,
                               const int src_is_float, const int dst_is_float,
                               const int has_alpha, const int need_clamp, const int need_norm,
                               const float alpha, const float min_v, const float max_v,
                               const float mean0, const float mean1, const float mean2,
                               const float std0, const float std1, const float std2) {
      const size_t dx = get_global_id(0);
      const size_t dy = get_global_id(1);

      if (dx < w && dy < h) {
          const size_t pixel = dy * w + dx;
          const size_t hw = w * h;
          const size_t src_base = pixel * c;
          for (size_t ch = 0; ch < c; ++ch) {
              float value = src_is_float ? ((__global const float*)src)[src_base + ch] : (float)src[src_base + ch];
              if (has_alpha) {
                  value *= alpha;
              }
              if (need_norm) {
                  const float mean_v = (ch < 2) ? ((ch == 0) ? mean0 : mean1) : mean2;
                  const float std_v = (ch < 2) ? ((ch == 0) ? std0 : std1) : std2;
                  value = (value - mean_v) / std_v;
              }
              if (need_clamp) {
                  value = (value < min_v) ? min_v : ((max_v < value) ? max_v : value);
              }
              const size_t dst_index = pixel + ch * hw;
              if (dst_is_float) {
                  ((__global float*)dst)[dst_index] = value;
              }
              else {
                  dst[dst_index] = (uchar)value;
              }
          }
      }
  }

   // CHW -> HWC
   __kernel void opencl_chw2hwc(const ulong c, const ulong h, const ulong w,
                                __global const uchar* src, __global uchar* dst,
                                const int src_is_float, const int dst_is_float,
                                const int has_alpha, const int need_clamp,
                                const float alpha, const float min_v, const float max_v) {
       const size_t dx = get_global_id(0);
       const size_t dy = get_global_id(1);

       if (dx < w && dy < h) {
           const size_t pixel = dy * w + dx;
           const size_t hw = w * h;
           const size_t dst_base = pixel * c;
           for (size_t ch = 0; ch < c; ++ch) {
               float value = src_is_float ? ((__global const float*)src)[pixel + ch * hw] : (float)src[pixel + ch * hw];
               if (has_alpha) {
                   value *= alpha;
               }
               if (need_clamp) {
                   value = (value < min_v) ? min_v : ((max_v < value) ? max_v : value);
               }
               if (dst_is_float) {
                   ((__global float*)dst)[dst_base + ch] = value;
               }
               else {
                   dst[dst_base + ch] = (uchar)value;
               }
           }
       }
   }

)";

enum struct InitOpenCLStatusEnum : int
    {
        Ready = 0,
        Inited = 1,
        Failed = 2,
    };
    class opencl {
    private:
        opencl() {
            static bool init0([]() {
                return initAll();
                }());
        }
    public:
        ~opencl() = default;
        opencl(const opencl&) = delete;
        opencl& operator=(const opencl&) = delete;
        opencl(opencl&&) = delete;
        opencl& operator=(opencl&&) = delete;
    public:
        static bool init() { return initAll(); }
        static bool release() { return releaseAll(); }

        // Query initialization state and the last backend error without
        // requiring library-side terminal diagnostics.
        static InitOpenCLStatusEnum status() { return initOpenCLStatus.load(std::memory_order_acquire); }
        static std::string lastError()
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            return lastOpenCLErrorStr;
        }

    public:
        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @tparam NeedNormalizedMeanStds input mean and stds are requires participation in calculations
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        * @param mean Array of mean values for normalization
        * @param stds Array of standard deviation values for normalization
        *
        * Normalization parameters are RGB-only.  A fourth channel reuses the
        * final parameter set to avoid reading outside the arrays.
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false,
                  bool NeedNormalizedMeanStds = false>
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f },
            const std::array<float, 3> stds = { 0.229f, 0.224f, 0.225f }) {
            opencl();
            if (h == 0 || w == 0 || c == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            if constexpr (!isSupportedElementType<Stype, Dtype>()) {
                // use cpu (the GPU kernels cover uint8_t and float element types)
                cpu::hwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds); return;
            }
            else {
                if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                    // use cpu
                    cpu::hwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds); return;
                }
                std::lock_guard<std::mutex> lock(OpenCLMutex);
                // use opencl
                const size_t pixel_size = h * w * c;
                const size_t input_size = pixel_size * sizeof(Stype);
                const size_t output_size = pixel_size * sizeof(Dtype);
                cl_mem opencl_input_memory = nullptr;
                cl_mem opencl_output_memory = nullptr;
                // alloc device memory
                cl_int clRes0 = CL_SUCCESS;
                opencl_input_memory = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, nullptr, &clRes0);
                cl_int clRes1 = CL_SUCCESS;
                opencl_output_memory = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &clRes1);
                if (clRes0 != CL_SUCCESS || clRes1 != CL_SUCCESS) {
                    if (opencl_input_memory != nullptr) { clReleaseMemObject(opencl_input_memory); }
                    if (opencl_output_memory != nullptr) { clReleaseMemObject(opencl_output_memory); }
                    fallbackToCpuHwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds, "OpenCL hwc2chw GPU execution failed"); return;
                }
                // copy host memory to device memory
                cl_int clRes2 = clEnqueueWriteBuffer(openclqueue, opencl_input_memory, CL_TRUE, 0, input_size, src, 0, nullptr, nullptr);
                if (clRes2 != CL_SUCCESS) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuHwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds, "OpenCL hwc2chw GPU execution failed"); return;
                }
                // call opencl function
                const bool launch_ok = runHwc2chwKernel(h, w, c, opencl_input_memory, opencl_output_memory,
                    isFloatElementType<Stype>(), isFloatElementType<Dtype>(),
                    HasAlpha, NeedClamp, NeedNormalizedMeanStds,
                    (float)alpha, (float)min_v, (float)max_v, mean, stds);
                if (!launch_ok) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuHwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds, "OpenCL hwc2chw GPU execution failed"); return;
                }
                // copy device memory to host memory; the blocking read also
                // guarantees the kernel has finished before the function returns
                cl_int clRes4 = clEnqueueReadBuffer(openclqueue, opencl_output_memory, CL_TRUE, 0, output_size, dst, 0, nullptr, nullptr);
                if (clRes4 != CL_SUCCESS) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuHwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds, "OpenCL hwc2chw GPU execution failed"); return;
                }
                clReleaseMemObject(opencl_input_memory);
                clReleaseMemObject(opencl_output_memory);
                clearCallFailures();
                return;
            }
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false>
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0, const Dtype max_v = 255) {
            opencl();
            if (c == 0 || h == 0 || w == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            if constexpr (!isSupportedElementType<Stype, Dtype>()) {
                // use cpu (the GPU kernels cover uint8_t and float element types)
                cpu::chw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v); return;
            }
            else {
                if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                    // use cpu
                    cpu::chw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v); return;
                }
                std::lock_guard<std::mutex> lock(OpenCLMutex);
                // use opencl
                const size_t pixel_size = h * w * c;
                const size_t input_size = pixel_size * sizeof(Stype);
                const size_t output_size = pixel_size * sizeof(Dtype);
                cl_mem opencl_input_memory = nullptr;
                cl_mem opencl_output_memory = nullptr;
                // alloc device memory
                cl_int clRes0 = CL_SUCCESS;
                opencl_input_memory = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, nullptr, &clRes0);
                cl_int clRes1 = CL_SUCCESS;
                opencl_output_memory = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &clRes1);
                if (clRes0 != CL_SUCCESS || clRes1 != CL_SUCCESS) {
                    if (opencl_input_memory != nullptr) { clReleaseMemObject(opencl_input_memory); }
                    if (opencl_output_memory != nullptr) { clReleaseMemObject(opencl_output_memory); }
                    fallbackToCpuChw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v, "OpenCL chw2hwc GPU execution failed"); return;
                }
                // copy host memory to device memory
                cl_int clRes2 = clEnqueueWriteBuffer(openclqueue, opencl_input_memory, CL_TRUE, 0, input_size, src, 0, nullptr, nullptr);
                if (clRes2 != CL_SUCCESS) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuChw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v, "OpenCL chw2hwc GPU execution failed"); return;
                }
                // call opencl function
                const bool launch_ok = runChw2hwcKernel(c, h, w, opencl_input_memory, opencl_output_memory,
                    isFloatElementType<Stype>(), isFloatElementType<Dtype>(),
                    HasAlpha, NeedClamp,
                    (float)alpha, (float)min_v, (float)max_v);
                if (!launch_ok) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuChw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v, "OpenCL chw2hwc GPU execution failed"); return;
                }
                // copy device memory to host memory; the blocking read also
                // guarantees the kernel has finished before the function returns
                cl_int clRes4 = clEnqueueReadBuffer(openclqueue, opencl_output_memory, CL_TRUE, 0, output_size, dst, 0, nullptr, nullptr);
                if (clRes4 != CL_SUCCESS) {
                    clReleaseMemObject(opencl_input_memory);
                    clReleaseMemObject(opencl_output_memory);
                    fallbackToCpuChw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v, "OpenCL chw2hwc GPU execution failed"); return;
                }
                clReleaseMemObject(opencl_input_memory);
                clReleaseMemObject(opencl_output_memory);
                clearCallFailures();
                return;
            }
        }


        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @tparam NeedNormalizedMeanStds input mean and stds are requires participation in calculations
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src OpenCL Memory (uint8_t) Pointer to the source data in HWC format
        * @param dst OpenCL Memory (float) Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        * @param mean Array of mean values for normalization
        * @param stds Array of standard deviation values for normalization
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false,
                  bool NeedNormalizedMeanStds = false>
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const cl_mem src, const cl_mem dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f },
            const std::array<float, 3> stds = { 0.229f, 0.224f, 0.225f }) {
            opencl();
            if (h == 0 || w == 0 || c == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            if constexpr (!isSupportedElementType<Stype, Dtype>()) {
                setLastError("OpenCL device-memory hwc2chw requires uint8_t/float element types.");
                return;
            }
            else {
                if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                    setLastError("OpenCL device-memory hwc2chw called before successful initialization.");
                    return;
                }
                std::lock_guard<std::mutex> lock(OpenCLMutex);
                const bool launch_ok = runHwc2chwKernel(h, w, c, src, dst,
                    isFloatElementType<Stype>(), isFloatElementType<Dtype>(),
                    HasAlpha, NeedClamp, NeedNormalizedMeanStds,
                    (float)alpha, (float)min_v, (float)max_v, mean, stds);
                if (!launch_ok) {
                    recordCallFailure("OpenCL device-memory hwc2chw launch failed");
                    return;
                }
                cl_int clRes = clFinish(openclqueue);
                if (clRes != CL_SUCCESS) {
                    recordCallFailure("OpenCL device-memory hwc2chw synchronization failed");
                    return;
                }
                clearCallFailures();
                return;
            }
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src OpenCL Memory (float) Pointer to the source data in CHW format
        * @param dst OpenCL Memory (uint8_t) Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false>
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const cl_mem src, const cl_mem dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0, const Dtype max_v = 255) {
            opencl();
            if (c == 0 || h == 0 || w == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            if constexpr (!isSupportedElementType<Stype, Dtype>()) {
                setLastError("OpenCL device-memory chw2hwc requires uint8_t/float element types.");
                return;
            }
            else {
                if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                    setLastError("OpenCL device-memory chw2hwc called before successful initialization.");
                    return;
                }
                std::lock_guard<std::mutex> lock(OpenCLMutex);
                const bool launch_ok = runChw2hwcKernel(c, h, w, src, dst,
                    isFloatElementType<Stype>(), isFloatElementType<Dtype>(),
                    HasAlpha, NeedClamp,
                    (float)alpha, (float)min_v, (float)max_v);
                if (!launch_ok) {
                    recordCallFailure("OpenCL device-memory chw2hwc launch failed");
                    return;
                }
                cl_int clRes = clFinish(openclqueue);
                if (clRes != CL_SUCCESS) {
                    recordCallFailure("OpenCL device-memory chw2hwc synchronization failed");
                    return;
                }
                clearCallFailures();
                return;
            }
        }

        /**
        * @brief Creates an OpenCL device buffer (helper for device-memory usage)
        *
        * @param bytes Size of the buffer in bytes
        * @param buffer Output cl_mem handle
        * @return true on success, false otherwise
        */
        static bool createDeviceBuffer(const size_t bytes, cl_mem* buffer) {
            opencl();
            if (bytes == 0 || buffer == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                return false;
            }
            cl_int clRes = CL_SUCCESS;
            cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &clRes);
            if (clRes != CL_SUCCESS || memory == nullptr) {
                setLastError("clCreateBuffer failed with error " + std::to_string(clRes) + ".");
                return false;
            }
            *buffer = memory;
            return true;
        }

        /**
        * @brief Destroys an OpenCL buffer created by createDeviceBuffer
        *
        * @param buffer cl_mem handle
        */
        static void destroyDeviceBuffer(const cl_mem buffer) {
            opencl();
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                return;
            }
            if (buffer != nullptr) {
                clReleaseMemObject(buffer);
            }
        }

        /**
        * @brief Uploads host data into an existing device buffer
        *
        * @param buffer OpenCL device buffer handle (created by createDeviceBuffer)
        * @param bytes Number of bytes to upload (may be smaller than the allocation)
        * @param src Pointer to the host source data
        * @return true on success, false otherwise
        */
        static bool uploadToDeviceBuffer(const cl_mem buffer, const size_t bytes, const void* src) {
            opencl();
            if (buffer == nullptr || bytes == 0 || src == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                return false;
            }
            const cl_int clRes = clEnqueueWriteBuffer(openclqueue, buffer, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr);
            if (clRes != CL_SUCCESS) {
                recordCallFailure("OpenCL device-buffer upload failed");
                return false;
            }
            clearCallFailures();
            return true;
        }

        /**
        * @brief Downloads data from an existing device buffer to host memory
        *
        * @param buffer OpenCL device buffer handle (created by createDeviceBuffer)
        * @param bytes Number of bytes to download (may be smaller than the allocation)
        * @param dst Pointer to the host destination data
        * @return true on success, false otherwise
        */
        static bool downloadFromDeviceBuffer(const cl_mem buffer, const size_t bytes, void* dst) {
            opencl();
            if (buffer == nullptr || bytes == 0 || dst == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited) {
                return false;
            }
            const cl_int clRes = clEnqueueReadBuffer(openclqueue, buffer, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr);
            if (clRes != CL_SUCCESS) {
                recordCallFailure("OpenCL device-buffer download failed");
                return false;
            }
            clearCallFailures();
            return true;
        }

    private:
        // The GPU kernels cover uint8_t and float element types; every other
        // Stype/Dtype combination keeps full CPU behavior via the fallback path.
        template <typename Stype, typename Dtype>
        static constexpr bool isSupportedElementType() {
            return (std::is_same<Stype, uint8_t>::value || std::is_same<Stype, float>::value) &&
                   (std::is_same<Dtype, uint8_t>::value || std::is_same<Dtype, float>::value);
        }

        template <typename Type>
        static constexpr bool isFloatElementType() {
            return std::is_same<Type, float>::value;
        }

        static bool runHwc2chwKernel(
            const size_t h, const size_t w, const size_t c,
            const cl_mem src, const cl_mem dst,
            const bool src_is_float, const bool dst_is_float,
            const bool has_alpha, const bool need_clamp, const bool need_norm,
            const float alpha, const float min_v, const float max_v,
            const std::array<float, 3>& mean, const std::array<float, 3>& stds)
        {
            if (hwc2chwOpenCLFun == nullptr) {
                return false;
            }
            const cl_ulong arg_h_val = (cl_ulong)h;
            const cl_ulong arg_w_val = (cl_ulong)w;
            const cl_ulong arg_c_val = (cl_ulong)c;
            const cl_mem arg_src_val = src;
            const cl_mem arg_dst_val = dst;
            const cl_int arg_src_is_float_val = src_is_float ? 1 : 0;
            const cl_int arg_dst_is_float_val = dst_is_float ? 1 : 0;
            const cl_int arg_has_alpha_val = has_alpha ? 1 : 0;
            const cl_int arg_need_clamp_val = need_clamp ? 1 : 0;
            const cl_int arg_need_norm_val = need_norm ? 1 : 0;
            const cl_float arg_alpha_val = alpha;
            const cl_float arg_min_v_val = min_v;
            const cl_float arg_max_v_val = max_v;
            const cl_float arg_mean_vals[3] = { mean[0], mean[1], mean[2] };
            const cl_float arg_std_vals[3] = { stds[0], stds[1], stds[2] };
            // for ready opencl kernel function(opencl_hwc2chw)
            cl_int clRes = CL_SUCCESS;
            const auto set_arg = [&](const cl_uint index, const size_t size, const void* value) {
                if (clRes == CL_SUCCESS) {
                    clRes = clSetKernelArg(hwc2chwOpenCLFun, index, size, value);
                }
            };
            set_arg(0, sizeof(cl_ulong), &arg_h_val);
            set_arg(1, sizeof(cl_ulong), &arg_w_val);
            set_arg(2, sizeof(cl_ulong), &arg_c_val);
            set_arg(3, sizeof(cl_mem), &arg_src_val);
            set_arg(4, sizeof(cl_mem), &arg_dst_val);
            set_arg(5, sizeof(cl_int), &arg_src_is_float_val);
            set_arg(6, sizeof(cl_int), &arg_dst_is_float_val);
            set_arg(7, sizeof(cl_int), &arg_has_alpha_val);
            set_arg(8, sizeof(cl_int), &arg_need_clamp_val);
            set_arg(9, sizeof(cl_int), &arg_need_norm_val);
            set_arg(10, sizeof(cl_float), &arg_alpha_val);
            set_arg(11, sizeof(cl_float), &arg_min_v_val);
            set_arg(12, sizeof(cl_float), &arg_max_v_val);
            set_arg(13, sizeof(cl_float), &arg_mean_vals[0]);
            set_arg(14, sizeof(cl_float), &arg_mean_vals[1]);
            set_arg(15, sizeof(cl_float), &arg_mean_vals[2]);
            set_arg(16, sizeof(cl_float), &arg_std_vals[0]);
            set_arg(17, sizeof(cl_float), &arg_std_vals[1]);
            set_arg(18, sizeof(cl_float), &arg_std_vals[2]);
            if (clRes != CL_SUCCESS) {
                return false;
            }
            const size_t global_size[2] = { w, h };
            clRes = clEnqueueNDRangeKernel(openclqueue, hwc2chwOpenCLFun, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
            return clRes == CL_SUCCESS;
        }

        static bool runChw2hwcKernel(
            const size_t c, const size_t h, const size_t w,
            const cl_mem src, const cl_mem dst,
            const bool src_is_float, const bool dst_is_float,
            const bool has_alpha, const bool need_clamp,
            const float alpha, const float min_v, const float max_v)
        {
            if (chw2hwcOpenCLFun == nullptr) {
                return false;
            }
            const cl_ulong arg_c_val = (cl_ulong)c;
            const cl_ulong arg_h_val = (cl_ulong)h;
            const cl_ulong arg_w_val = (cl_ulong)w;
            const cl_mem arg_src_val = src;
            const cl_mem arg_dst_val = dst;
            const cl_int arg_src_is_float_val = src_is_float ? 1 : 0;
            const cl_int arg_dst_is_float_val = dst_is_float ? 1 : 0;
            const cl_int arg_has_alpha_val = has_alpha ? 1 : 0;
            const cl_int arg_need_clamp_val = need_clamp ? 1 : 0;
            const cl_float arg_alpha_val = alpha;
            const cl_float arg_min_v_val = min_v;
            const cl_float arg_max_v_val = max_v;
            // for ready opencl kernel function(opencl_chw2hwc)
            cl_int clRes = CL_SUCCESS;
            const auto set_arg = [&](const cl_uint index, const size_t size, const void* value) {
                if (clRes == CL_SUCCESS) {
                    clRes = clSetKernelArg(chw2hwcOpenCLFun, index, size, value);
                }
            };
            set_arg(0, sizeof(cl_ulong), &arg_c_val);
            set_arg(1, sizeof(cl_ulong), &arg_h_val);
            set_arg(2, sizeof(cl_ulong), &arg_w_val);
            set_arg(3, sizeof(cl_mem), &arg_src_val);
            set_arg(4, sizeof(cl_mem), &arg_dst_val);
            set_arg(5, sizeof(cl_int), &arg_src_is_float_val);
            set_arg(6, sizeof(cl_int), &arg_dst_is_float_val);
            set_arg(7, sizeof(cl_int), &arg_has_alpha_val);
            set_arg(8, sizeof(cl_int), &arg_need_clamp_val);
            set_arg(9, sizeof(cl_float), &arg_alpha_val);
            set_arg(10, sizeof(cl_float), &arg_min_v_val);
            set_arg(11, sizeof(cl_float), &arg_max_v_val);
            if (clRes != CL_SUCCESS) {
                return false;
            }
            const size_t global_size[2] = { w, h };
            clRes = clEnqueueNDRangeKernel(openclqueue, chw2hwcOpenCLFun, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
            return clRes == CL_SUCCESS;
        }

        // GPU and CPU fallback kernels clamp float->uint8 values before the
        // narrow.  This avoids undefined behavior and keeps results identical.
        template <typename Stype, typename Dtype,
                  bool HasAlpha, bool NeedClamp, bool NeedNormalizedMeanStds>
        static void fallbackToCpuHwc2chw(size_t h, size_t w, size_t c,
            const Stype* src, Dtype* dst, const Dtype alpha,
            const Dtype min_v, const Dtype max_v,
            const std::array<float, 3>& mean, const std::array<float, 3>& stds,
            const char* message)
        {
            recordCallFailure(message);
            cpu::hwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(h, w, c, src, dst, alpha, min_v, max_v, mean, stds);
        }

        template <typename Stype, typename Dtype,
                  bool HasAlpha, bool NeedClamp>
        static void fallbackToCpuChw2hwc(size_t c, size_t h, size_t w,
            const Stype* src, Dtype* dst, const Dtype alpha,
            const Dtype min_v, const Dtype max_v,
            const char* message)
        {
            recordCallFailure(message);
            cpu::chw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(c, h, w, src, dst, alpha, min_v, max_v);
        }

        static void setLastError(const std::string& message)
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            lastOpenCLErrorStr = message;
        }

        // Repeated transient failures stop repeated expensive GPU attempts.
        // The backend remains available after release()/init() resets state.
        static void recordCallFailure(const std::string& message)
        {
            const auto failures = ++consecutiveOpenCLFailures;
            const std::string fullMessage = message + " (consecutive OpenCL failures: " +
                std::to_string(failures) + ").";
            setLastError(fullMessage);

            if (failures >= maxConsecutiveOpenCLFailures)
            {
                initOpenCLStatus.store(InitOpenCLStatusEnum::Failed, std::memory_order_release);
                setLastError(fullMessage + " OpenCL backend disabled until release()/init().");
            }
        }

        static void clearCallFailures()
        {
            consecutiveOpenCLFailures.store(0, std::memory_order_release);
        }

        static std::string findOpenCLModuleName()
        {
#ifdef _WIN32
            // The Windows ICD loader ships with the system (OpenCL.dll).
            return "OpenCL.dll";
#elif defined(__APPLE__)
            // macOS ships the OpenCL framework as part of the system, and the
            // framework binary exports the cl* entry points itself.
            return "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
            // The ICD loader soname every OpenCL implementation provides.
            return "libOpenCL.so.1";
#endif
        }

        static bool initOpenCLDriverAPI()
        {
            const std::string opencl_lib = findOpenCLModuleName();
            opencl_library_name = opencl_lib;
            auto* dlManager = whyb::DynamicLibraryManager::instance();
            auto openclDriver = dlManager->loadLibrary(opencl_lib);
            if (!openclDriver)
            {
                setLastError("Failed to load OpenCL driver library: " + opencl_lib + ".");
                return false;
            }
            clGetPlatformIDs = (clGetPlatformIDs_t)(dlManager->getFunction(opencl_lib, "clGetPlatformIDs"));
            clGetPlatformInfo = (clGetPlatformInfo_t)(dlManager->getFunction(opencl_lib, "clGetPlatformInfo"));
            clGetDeviceIDs = (clGetDeviceIDs_t)(dlManager->getFunction(opencl_lib, "clGetDeviceIDs"));
            clGetDeviceInfo = (clGetDeviceInfo_t)(dlManager->getFunction(opencl_lib, "clGetDeviceInfo"));
            clCreateContext = (clCreateContext_t)(dlManager->getFunction(opencl_lib, "clCreateContext"));
            clReleaseContext = (clReleaseContext_t)(dlManager->getFunction(opencl_lib, "clReleaseContext"));
            clCreateCommandQueue = (clCreateCommandQueue_t)(dlManager->getFunction(opencl_lib, "clCreateCommandQueue"));
            clReleaseCommandQueue = (clReleaseCommandQueue_t)(dlManager->getFunction(opencl_lib, "clReleaseCommandQueue"));
            clCreateBuffer = (clCreateBuffer_t)(dlManager->getFunction(opencl_lib, "clCreateBuffer"));
            clReleaseMemObject = (clReleaseMemObject_t)(dlManager->getFunction(opencl_lib, "clReleaseMemObject"));
            clCreateProgramWithSource = (clCreateProgramWithSource_t)(dlManager->getFunction(opencl_lib, "clCreateProgramWithSource"));
            clBuildProgram = (clBuildProgram_t)(dlManager->getFunction(opencl_lib, "clBuildProgram"));
            clGetProgramBuildInfo = (clGetProgramBuildInfo_t)(dlManager->getFunction(opencl_lib, "clGetProgramBuildInfo"));
            clReleaseProgram = (clReleaseProgram_t)(dlManager->getFunction(opencl_lib, "clReleaseProgram"));
            clCreateKernel = (clCreateKernel_t)(dlManager->getFunction(opencl_lib, "clCreateKernel"));
            clReleaseKernel = (clReleaseKernel_t)(dlManager->getFunction(opencl_lib, "clReleaseKernel"));
            clSetKernelArg = (clSetKernelArg_t)(dlManager->getFunction(opencl_lib, "clSetKernelArg"));
            clEnqueueNDRangeKernel = (clEnqueueNDRangeKernel_t)(dlManager->getFunction(opencl_lib, "clEnqueueNDRangeKernel"));
            clEnqueueReadBuffer = (clEnqueueReadBuffer_t)(dlManager->getFunction(opencl_lib, "clEnqueueReadBuffer"));
            clEnqueueWriteBuffer = (clEnqueueWriteBuffer_t)(dlManager->getFunction(opencl_lib, "clEnqueueWriteBuffer"));
            clFinish = (clFinish_t)(dlManager->getFunction(opencl_lib, "clFinish"));

            if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo ||
                !clCreateContext || !clReleaseContext || !clCreateCommandQueue || !clReleaseCommandQueue ||
                !clCreateBuffer || !clReleaseMemObject || !clCreateProgramWithSource || !clBuildProgram ||
                !clGetProgramBuildInfo || !clReleaseProgram || !clCreateKernel || !clReleaseKernel ||
                !clSetKernelArg || !clEnqueueNDRangeKernel || !clEnqueueReadBuffer || !clEnqueueWriteBuffer ||
                !clFinish) {
                setLastError("Failed to load one or more OpenCL functions from: " + opencl_lib + ".");
                return false;
            }
            return true;
        }

        static bool selectOpenCLDevice(const std::vector<cl_platform_id>& platforms,
            cl_platform_id& selected_platform, cl_device_id& selected_device)
        {
            // Prefer the GPU device (strongest compute) first, then the device
            // with more compute units, then the higher clock frequency.
            int best_rank = -1;
            cl_uint best_units = 0;
            cl_uint best_clock = 0;
            bool found = false;
            for (const auto& current_platform : platforms)
            {
                cl_uint device_count = 0;
                cl_int clRes = clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
                if (clRes != CL_SUCCESS || device_count == 0)
                {
                    continue;
                }
                std::vector<cl_device_id> devices(device_count);
                clRes = clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
                if (clRes != CL_SUCCESS)
                {
                    continue;
                }
                for (const auto& current_device : devices)
                {
                    cl_device_type device_type = 0;
                    cl_uint compute_units = 0;
                    cl_uint clock_frequency = 0;
                    clGetDeviceInfo(current_device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
                    clGetDeviceInfo(current_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
                    clGetDeviceInfo(current_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, nullptr);

                    const int rank = ((device_type & CL_DEVICE_TYPE_GPU) != 0) ? 2
                                   : ((device_type & CL_DEVICE_TYPE_ACCELERATOR) != 0) ? 1 : 0;
                    if (!found || rank > best_rank ||
                        (rank == best_rank && compute_units > best_units) ||
                        (rank == best_rank && compute_units == best_units && clock_frequency > best_clock))
                    {
                        best_rank = rank;
                        best_units = compute_units;
                        best_clock = clock_frequency;
                        selected_platform = current_platform;
                        selected_device = current_device;
                        found = true;
                    }
                }
            }
            if (!found)
            {
                setLastError("No OpenCL device found.");
                return false;
            }
            return true;
        }

        static bool initOpenCLFunctions()
        {
            cl_int clRes = CL_SUCCESS;
            cl_uint num_platforms = 0;
            clRes = clGetPlatformIDs(0, nullptr, &num_platforms);
            if (clRes != CL_SUCCESS || num_platforms == 0)
            {
                setLastError("clGetPlatformIDs failed with error " + std::to_string(clRes) + ".");
                return false;
            }
            std::vector<cl_platform_id> platforms(num_platforms);
            clRes = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            if (clRes != CL_SUCCESS)
            {
                setLastError("clGetPlatformIDs failed with error " + std::to_string(clRes) + ".");
                return false;
            }

            if (!selectOpenCLDevice(platforms, platform, device))
            {
                return false;
            }

            const cl_context_properties context_properties[] = {
                (cl_context_properties)CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
            };
            context = clCreateContext(context_properties, 1, &device, nullptr, nullptr, &clRes);
            if (clRes != CL_SUCCESS || context == nullptr)
            {
                setLastError("clCreateContext failed with error " + std::to_string(clRes) + ".");
                return false;
            }
            openclqueue = clCreateCommandQueue(context, device, 0, &clRes); //flags: 0 = in-order execution
            if (clRes != CL_SUCCESS || openclqueue == nullptr)
            {
                setLastError("clCreateCommandQueue failed with error " + std::to_string(clRes) + ".");
                clReleaseContext(context);
                context = nullptr;
                return false;
            }

            // Load OpenCL C source code to device program
            const char* source_str = openclSource;
            openclprogram = clCreateProgramWithSource(context, 1, &source_str, nullptr, &clRes);
            if (clRes != CL_SUCCESS || openclprogram == nullptr)
            {
                setLastError("clCreateProgramWithSource failed with error " + std::to_string(clRes) + ".");
                clReleaseCommandQueue(openclqueue);
                openclqueue = nullptr;
                clReleaseContext(context);
                context = nullptr;
                return false;
            }

            // compile OpenCL C source code
            clRes = clBuildProgram(openclprogram, 1, &device, nullptr, nullptr, nullptr);
            if (clRes != CL_SUCCESS)
            {
                size_t logSize = 0;
                clGetProgramBuildInfo(openclprogram, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                std::string log(logSize, '\0');
                if (logSize > 0)
                {
                    clGetProgramBuildInfo(openclprogram, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
                }
                setLastError("OpenCL Compile error: " + log);
                clReleaseProgram(openclprogram);
                openclprogram = nullptr;
                clReleaseCommandQueue(openclqueue);
                openclqueue = nullptr;
                clReleaseContext(context);
                context = nullptr;
                return false;
            }

            // Get opencl module kernel function(opencl_hwc2chw)
            hwc2chwOpenCLFun = clCreateKernel(openclprogram, "opencl_hwc2chw", &clRes);
            if (clRes != CL_SUCCESS || hwc2chwOpenCLFun == nullptr)
            {
                setLastError("clCreateKernel (opencl_hwc2chw) failed with error " + std::to_string(clRes) + ".");
                clReleaseProgram(openclprogram);
                openclprogram = nullptr;
                clReleaseCommandQueue(openclqueue);
                openclqueue = nullptr;
                clReleaseContext(context);
                context = nullptr;
                return false;
            }
            // Get opencl module kernel function(opencl_chw2hwc)
            chw2hwcOpenCLFun = clCreateKernel(openclprogram, "opencl_chw2hwc", &clRes);
            if (clRes != CL_SUCCESS || chw2hwcOpenCLFun == nullptr)
            {
                setLastError("clCreateKernel (opencl_chw2hwc) failed with error " + std::to_string(clRes) + ".");
                clReleaseKernel(hwc2chwOpenCLFun);
                hwc2chwOpenCLFun = nullptr;
                clReleaseProgram(openclprogram);
                openclprogram = nullptr;
                clReleaseCommandQueue(openclqueue);
                openclqueue = nullptr;
                clReleaseContext(context);
                context = nullptr;
                return false;
            }
            return true;
        }

        static bool initAll()
        {
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            setLastError("");
            if (initOpenCLStatus.load(std::memory_order_acquire) == InitOpenCLStatusEnum::Ready) {
                if (!initOpenCLDriverAPI()) {
                    setLastError("Failed to load OpenCL driver API functions.");
                    lastOpenCLErrorStr = "Failed to load OpenCL driver API functions.";
                    initOpenCLStatus.store(InitOpenCLStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                if (!initOpenCLFunctions()) {
                    setLastError("Failed to initialize the OpenCL device and kernels.");
                    lastOpenCLErrorStr = "Failed to initialize the OpenCL device and kernels.";
                    initOpenCLStatus.store(InitOpenCLStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                initOpenCLStatus.store(InitOpenCLStatusEnum::Inited, std::memory_order_release);
                return true;
            }
            else if (initOpenCLStatus.load(std::memory_order_acquire) == InitOpenCLStatusEnum::Inited) {
                return true;
            }
            else if (initOpenCLStatus.load(std::memory_order_acquire) == InitOpenCLStatusEnum::Failed) {
                setLastError("OpenCL initialization failed. " + lastError());
                return false;
            }
            return true;
        }

        static bool releaseAll()
        {
            std::lock_guard<std::mutex> lock(OpenCLMutex);
            if (initOpenCLStatus.load(std::memory_order_acquire) != InitOpenCLStatusEnum::Inited)
            {
                return true;
            }

            if (chw2hwcOpenCLFun != nullptr) {
                clReleaseKernel(chw2hwcOpenCLFun);
                chw2hwcOpenCLFun = nullptr;
            }
            if (hwc2chwOpenCLFun != nullptr) {
                clReleaseKernel(hwc2chwOpenCLFun);
                hwc2chwOpenCLFun = nullptr;
            }
            if (openclprogram != nullptr) {
                clReleaseProgram(openclprogram);
                openclprogram = nullptr;
            }
            if (openclqueue != nullptr) {
                clReleaseCommandQueue(openclqueue);
                openclqueue = nullptr;
            }
            if (context != nullptr) {
                clReleaseContext(context);
                context = nullptr;
            }
            auto* dlManager = whyb::DynamicLibraryManager::instance();
            if (!opencl_library_name.empty()) {
                dlManager->unloadLibrary(opencl_library_name);
                opencl_library_name.clear();
            }
            consecutiveOpenCLFailures.store(0, std::memory_order_release);
            initOpenCLStatus.store(InitOpenCLStatusEnum::Ready, std::memory_order_release);
            return true;
        }
    private:
        inline static std::atomic<InitOpenCLStatusEnum> initOpenCLStatus = InitOpenCLStatusEnum::Ready;
        inline static std::string lastOpenCLErrorStr = "";
        inline static std::mutex OpenCLMutex;
        inline static std::mutex errorMutex;
        inline static std::atomic<size_t> consecutiveOpenCLFailures = 0;
        static constexpr size_t maxConsecutiveOpenCLFailures = 3;

        inline static std::string opencl_library_name = "";

        inline static cl_platform_id platform = nullptr;
        inline static cl_device_id device = nullptr;
        inline static cl_context context = nullptr;
        inline static cl_command_queue openclqueue = nullptr;
        inline static cl_program openclprogram = nullptr;

        inline static cl_kernel hwc2chwOpenCLFun = nullptr;
        inline static cl_kernel chw2hwcOpenCLFun = nullptr;
    };

} // namespace whyb