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

// NVRTC enum type define
typedef enum {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;

// NVRTC function type define
typedef nvrtcResult(*nvrtcCreateProgram_t)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef nvrtcResult(*nvrtcCompileProgram_t)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult(*nvrtcGetPTXSize_t)(nvrtcProgram, size_t*);
typedef nvrtcResult(*nvrtcGetPTX_t)(nvrtcProgram, char*);
typedef nvrtcResult(*nvrtcDestroyProgram_t)(nvrtcProgram*);
typedef nvrtcResult(*nvrtcGetProgramLogSize_t)(nvrtcProgram, size_t*);
typedef nvrtcResult(*nvrtcGetProgramLog_t)(nvrtcProgram, char*);
typedef const char* (*nvrtcGetErrorString_t)(nvrtcResult);

// NVIDIA CUDA Driver API data type define
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

// NVIDIA CUDA Driver API function type define
typedef CUresult(*cuInit_t)(unsigned int);
typedef CUresult(*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult(*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(*cuCtxDestroy_t)(CUcontext);
typedef CUresult(*cuStreamCreate_t)(CUstream*, unsigned int);
typedef CUresult(*cuStreamDestroy_t)(CUstream);
typedef CUresult(*cuStreamSynchronize_t)(CUstream);
typedef CUresult(*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult(*cuModuleUnload_t)(CUmodule);
typedef CUresult(*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult(*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUcontext, void**, void**);
typedef CUresult(*cuCtxSynchronize_t)(void);

typedef CUresult(*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult(*cuMemAllocHost_t)(void**, size_t);
typedef CUresult(*cuMemFree_t)(CUdeviceptr);
typedef CUresult(*cuMemFreeHost_t)(void*);

typedef CUresult(*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult(*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);


#ifdef _WIN32
#define DYNAMIC_LIBRARY_EXTENSION ".dll"
#else
#define DYNAMIC_LIBRARY_EXTENSION ".so"
#endif

// CUDA Driver API function pointers. These are inline variables (one shared
// object across translation units) instead of file-scope statics, so every TU
// that includes this header observes the same initialized set.
inline cuInit_t cuInit = nullptr;
inline cuDeviceGet_t cuDeviceGet = nullptr;
inline cuCtxCreate_t cuCtxCreate = nullptr;
inline cuCtxDestroy_t cuCtxDestroy = nullptr;
inline cuStreamCreate_t cuStreamCreate = nullptr;
inline cuStreamDestroy_t cuStreamDestroy = nullptr;
inline cuStreamSynchronize_t cuStreamSynchronize = nullptr;
inline cuModuleLoadDataEx_t cuModuleLoadDataEx = nullptr;
inline cuModuleUnload_t cuModuleUnload = nullptr;
inline cuModuleGetFunction_t cuModuleGetFunction = nullptr;
inline cuLaunchKernel_t cuLaunchKernel = nullptr;
inline cuCtxSynchronize_t cuCtxSynchronize = nullptr;
inline cuMemAlloc_t cuMemAlloc = nullptr;
inline cuMemAllocHost_t cuMemAllocHost = nullptr;
inline cuMemFree_t cuMemFree = nullptr;
inline cuMemFreeHost_t cuMemFreeHost = nullptr;
inline cuMemcpyHtoD_t cuMemcpyHtoD = nullptr;
inline cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;

static const char* cudaSource = R"(
  typedef unsigned char uint8_t;

  // HWC -> CHW
  extern "C" __global__ void cuda_hwc2chw(const size_t h, const size_t w, const size_t c,
                                          const uint8_t* __restrict__ src, float* __restrict__ dst, const float alpha = 1.0f) {
      int dx = blockIdx.x * blockDim.x + threadIdx.x;
      int dy = blockIdx.y * blockDim.y + threadIdx.y;

      if ((size_t)dx < w && (size_t)dy < h) {
          const size_t pixel = (size_t)dy * w + (size_t)dx;
          const size_t src_base = pixel * c;
          for (size_t ch = 0; ch < c; ++ch) {
              dst[pixel + ch * w * h] = static_cast<float>(src[src_base + ch] * alpha);
          }
      }
  }

   // CHW -> HWC
   extern "C" __global__ void cuda_chw2hwc(const size_t c, const size_t h, const size_t w,
                                           const float* __restrict__ src, uint8_t* __restrict__ dst, const uint8_t alpha = 1) {
       int dx = blockIdx.x * blockDim.x + threadIdx.x;
       int dy = blockIdx.y * blockDim.y + threadIdx.y;

       if ((size_t)dx < w && (size_t)dy < h) {
           const size_t pixel = (size_t)dy * w + (size_t)dx;
           const size_t dst_base = pixel * c;
           for (size_t ch = 0; ch < c; ++ch) {
               float value = src[pixel + ch * w * h] * alpha;
               dst[dst_base + ch] = static_cast<uint8_t>(value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value));
           }
       }
   }

)";

enum struct InitCUDAStatusEnum : int
    {
        Ready = 0,
        Inited = 1,
        Failed = 2,
    };
    class nvidia {
    private:
        nvidia() {
            static bool init0([]() {
                return initAll();
                }());
        }
    public:
        ~nvidia() = default;
        nvidia(const nvidia&) = delete;
        nvidia& operator=(const nvidia&) = delete;
        nvidia(nvidia&&) = delete;
        nvidia& operator=(nvidia&&) = delete;
    public:
        static bool init() { return initAll(); }
        static bool release() { return releaseAll(); }

        // Query initialization state and the last backend error without
        // requiring library-side terminal diagnostics.
        static InitCUDAStatusEnum status() { return initCUDAStatus.load(std::memory_order_acquire); }
        static std::string lastError()
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            return lastCUDAErrorStr;
        }

    public:
        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        */
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const uint8_t* src, float* dst,
            const float alpha = 1.f / 255.f) {
            nvidia();
            if (initCUDAStatus.load(std::memory_order_acquire) != InitCUDAStatusEnum::Inited) {
                // use cpu
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha); return;
            }
            // use cuda
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(uint8_t);
            const size_t output_size = pixel_size * sizeof(float);
            CUdeviceptr cuda_input_memory = 0;
            CUdeviceptr cuda_output_memory = 0;
            // alloc device memory
            CUresult cuRes0 = cuMemAlloc(&cuda_input_memory, input_size);
            CUresult cuRes1 = cuMemAlloc(&cuda_output_memory, output_size);
            if (cuRes0 != 0 || cuRes1 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "CUDA hwc2chw GPU execution failed"); return;
            }
            // copy host memory to device memory
            CUresult cuRes2 = cuMemcpyHtoD(cuda_input_memory, src, input_size);
            if (cuRes2 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "CUDA hwc2chw GPU execution failed"); return;
            }
            // call cuda function
            if (hwc2chwCUDAFun == nullptr) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "CUDA hwc2chw GPU execution failed"); return;
            }
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            size_t arg_c_val = c;
            float arg_alpha_val = alpha;
            void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &cuda_input_memory, &cuda_output_memory, &arg_alpha_val };
            CUresult cuRes3 = cuLaunchKernel(
                hwc2chwCUDAFun, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, nullptr, args1, nullptr);
            if (cuRes3 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "CUDA hwc2chw GPU execution failed"); return;
            }
            // copy device memory to host memory; the synchronous copy also
            // guarantees the kernel has finished before the function returns
            CUresult cuRes4 = cuMemcpyDtoH(dst, cuda_output_memory, output_size);
            if (cuRes4 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "CUDA hwc2chw GPU execution failed"); return;
            }
            cuMemFree(cuda_input_memory);
            cuMemFree(cuda_output_memory);
            clearCallFailures();
            return;
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const float* src, uint8_t* dst,
            const uint8_t alpha = 255.0f) {
            nvidia();
            if (initCUDAStatus.load(std::memory_order_acquire) != InitCUDAStatusEnum::Inited) {
                // use cpu
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha); return;
            }
            // use cuda
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(float);
            const size_t output_size = pixel_size * sizeof(uint8_t);
            CUdeviceptr cuda_input_memory = 0;
            CUdeviceptr cuda_output_memory = 0;
            // alloc device memory
            CUresult cuRes0 = cuMemAlloc(&cuda_input_memory, input_size);
            CUresult cuRes1 = cuMemAlloc(&cuda_output_memory, output_size);
            if (cuRes0 != 0 || cuRes1 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "CUDA chw2hwc GPU execution failed"); return;
            }
            // copy host memory to device memory
            CUresult cuRes2 = cuMemcpyHtoD(cuda_input_memory, src, input_size);
            if (cuRes2 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "CUDA chw2hwc GPU execution failed"); return;
            }
            // call cuda function
            if (chw2hwcCUDAFun == nullptr) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "CUDA chw2hwc GPU execution failed"); return;
            }
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_c_val = c;
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            uint8_t arg_alpha_val = alpha;
            void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &cuda_input_memory, &cuda_output_memory, &arg_alpha_val };
            CUresult cuRes3 = cuLaunchKernel(
                chw2hwcCUDAFun, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, nullptr, args, nullptr);
            if (cuRes3 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "CUDA chw2hwc GPU execution failed"); return;
            }
            // copy device memory to host memory; the synchronous copy also
            // guarantees the kernel has finished before the function returns
            CUresult cuRes4 = cuMemcpyDtoH(dst, cuda_output_memory, output_size);
            if (cuRes4 != 0) {
                cuMemFree(cuda_input_memory);
                cuMemFree(cuda_output_memory);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "CUDA chw2hwc GPU execution failed"); return;
            }
            cuMemFree(cuda_input_memory);
            cuMemFree(cuda_output_memory);
            clearCallFailures();
            return;
        }


        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Cuda Memory (uint8_t) Pointer to the source data in HWC format
        * @param dst Cuda Memory (float) Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        */
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            CUdeviceptr src, CUdeviceptr dst,
            const float alpha = 1.f / 255.f) {
            nvidia();
            if (initCUDAStatus.load(std::memory_order_acquire) != InitCUDAStatusEnum::Inited) {
                setLastError("CUDA device-memory hwc2chw called before successful initialization.");
                return;
            }
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(uint8_t);
            const size_t output_size = pixel_size * sizeof(float);

            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            size_t arg_c_val = c;
            float arg_alpha_val = alpha;
            void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &src, &dst, &arg_alpha_val };
            CUresult cuRes0 = cuLaunchKernel(
                hwc2chwCUDAFun, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, nullptr, args1, nullptr);
            if (cuRes0 != 0) {
                recordCallFailure("CUDA device-memory hwc2chw launch failed");
                return;
            }
            CUresult cuRes1 = cuCtxSynchronize();
            if (cuRes1 != 0) {
                recordCallFailure("CUDA device-memory hwc2chw synchronization failed");
                return;
            }
            clearCallFailures();
            return;
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Cuda Memory (float) Pointer to the source data in CHW format
        * @param dst Cuda Memory (uint8_t) Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            CUdeviceptr src, CUdeviceptr dst,
            const uint8_t alpha = 255.0f) {
            nvidia();
            if (initCUDAStatus.load(std::memory_order_acquire) != InitCUDAStatusEnum::Inited) {
                setLastError("CUDA device-memory chw2hwc called before successful initialization.");
                return;
            }
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_c_val = c;
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            uint8_t arg_alpha_val = alpha;
            void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &src, &dst, &arg_alpha_val };
            CUresult cuRes0 = cuLaunchKernel(
                chw2hwcCUDAFun, gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, nullptr, args, nullptr);
            if (cuRes0 != 0) {
                recordCallFailure("CUDA device-memory chw2hwc launch failed");
                return;
            }
            CUresult cuRes1 = cuCtxSynchronize();
            if (cuRes1 != 0) {
                recordCallFailure("CUDA device-memory chw2hwc synchronization failed");
                return;
            }
            clearCallFailures();
            return;
        }
    private:
        // GPU and CPU fallback kernels clamp float->uint8 values before the
        // narrow.  This avoids undefined behavior and keeps results identical.
        static void fallbackToCpuHwc2chw(size_t h, size_t w, size_t c,
            const uint8_t* src, float* dst, float alpha, const char* message)
        {
            recordCallFailure(message);
            cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
        }

        static void fallbackToCpuChw2hwc(size_t c, size_t h, size_t w,
            const float* src, uint8_t* dst, uint8_t alpha, const char* message)
        {
            recordCallFailure(message);
            cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha);
        }

        static void setLastError(const std::string& message)
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            lastCUDAErrorStr = message;
        }

        // Repeated transient failures stop repeated expensive GPU attempts.
        // The backend remains available after release()/init() resets state.
        static void recordCallFailure(const std::string& message)
        {
            const auto failures = ++consecutiveCUDAFailures;
            const std::string fullMessage = message + " (consecutive CUDA failures: " +
                std::to_string(failures) + ").";
            setLastError(fullMessage);

            if (failures >= maxConsecutiveCUDAFailures)
            {
                std::lock_guard<std::mutex> lock(CUDAMutex);
                initCUDAStatus.store(InitCUDAStatusEnum::Failed, std::memory_order_release);
                setLastError(fullMessage + " CUDA backend disabled until release()/init().");
            }
        }

        static void clearCallFailures()
        {
            consecutiveCUDAFailures.store(0, std::memory_order_release);
        }

        static std::string compileCUDAWithNVRTC(const std::string& libraryName, const std::string& cudaSource)
        {
            // dynamic load NVRTC lib
            auto* dlManager = DynamicLibraryManager::instance();
            auto nvrtcLib = dlManager->loadLibrary(libraryName);
            if (!nvrtcLib)
            {
                setLastError("Failed to load NVRTC library: " + libraryName + ".");
                return "";
            }

            // Get NVRTC function points
            auto nvrtcCreateProgram_fun = (nvrtcCreateProgram_t)(dlManager->getFunction(libraryName, "nvrtcCreateProgram"));
            auto nvrtcCompileProgram_fun = (nvrtcCompileProgram_t)(dlManager->getFunction(libraryName, "nvrtcCompileProgram"));
            auto nvrtcGetPTXSize_fun = (nvrtcGetPTXSize_t)(dlManager->getFunction(libraryName, "nvrtcGetPTXSize"));
            auto nvrtcGetPTX_fun = (nvrtcGetPTX_t)(dlManager->getFunction(libraryName, "nvrtcGetPTX"));
            auto nvrtcDestroyProgram_fun = (nvrtcDestroyProgram_t)(dlManager->getFunction(libraryName, "nvrtcDestroyProgram"));
            auto nvrtcGetProgramLogSize_fun = (nvrtcGetProgramLogSize_t)(dlManager->getFunction(libraryName, "nvrtcGetProgramLogSize"));
            auto nvrtcGetProgramLog_fun = (nvrtcGetProgramLog_t)(dlManager->getFunction(libraryName, "nvrtcGetProgramLog"));
            auto nvrtcGetErrorString_fun = (nvrtcGetErrorString_t)(dlManager->getFunction(libraryName, "nvrtcGetErrorString"));

            // Check function point is not nullptr
            if (!nvrtcCreateProgram_fun || !nvrtcCompileProgram_fun || !nvrtcGetPTXSize_fun ||
                !nvrtcGetPTX_fun || !nvrtcDestroyProgram_fun || !nvrtcGetProgramLogSize_fun ||
                !nvrtcGetProgramLog_fun || !nvrtcGetErrorString_fun)
            {
                setLastError("Failed to load NVRTC functions from: " + libraryName + ".");
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // Create NVRTC Program Object
            nvrtcProgram prog;
            nvrtcResult res = nvrtcCreateProgram_fun(&prog, cudaSource.c_str(), "FastChwHwcConverterCuda.cu", 0, nullptr, nullptr);
            if (res != NVRTC_SUCCESS)
            {
                setLastError(std::string("nvrtcCreateProgram failed: ") + nvrtcGetErrorString_fun(res));
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // compile CUDA source code
            const char* options[] = { "-default-device", "--std=c++11" };
            res = nvrtcCompileProgram_fun(prog, 2, options);
            if (res != NVRTC_SUCCESS)
            {
                size_t logSize;
                nvrtcGetProgramLogSize_fun(prog, &logSize);
                std::string log(logSize, '\0');
                nvrtcGetProgramLog_fun(prog, &log[0]);
                setLastError("CUDA Compile error: " + log);
                nvrtcDestroyProgram_fun(&prog);
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // Get PTX String
            size_t ptxSize;
            nvrtcGetPTXSize_fun(prog, &ptxSize);
            std::string ptx(ptxSize, '\0');
            nvrtcGetPTX_fun(prog, &ptx[0]);

            // Release NVRTC Program
            nvrtcDestroyProgram_fun(&prog);

            // Release library
            dlManager->unloadLibrary(libraryName);
            return ptx;
        }

        static std::string findNVRTCModuleName()
        {
#ifdef _WIN32
            char currentDir[MAX_PATH] = { 0 };
            if (GetModuleFileNameA(nullptr, currentDir, MAX_PATH) == 0)
            {
                setLastError("Failed to get current directory on Windows.");
                return "";
            }

            std::string executablePath(currentDir);
            auto lastSlash = executablePath.find_last_of("\\/");
            if (lastSlash != std::string::npos)
            {
                executablePath = executablePath.substr(0, lastSlash);
            }
#else
            char currentDir[PATH_MAX] = { 0 };
            if (readlink("/proc/self/exe", currentDir, PATH_MAX) == -1)
            {
                setLastError("Failed to get current directory on Linux.");
                return "";
            }

            std::string executablePath(currentDir);
            auto lastSlash = executablePath.find_last_of("/");
            if (lastSlash != std::string::npos)
            {
                executablePath = executablePath.substr(0, lastSlash);
            }
#endif
            // CUDA version list: 13.2 ~ 10.0
            const std::vector<std::string> cudaVersions = {
                // 13.x
                "132_0", "131_0", "130_0",
                // 12.x
                "129_0", "128_0", "127_0", "126_0", "125_0", "124_0", "123_0", "122_0", "121_0", "120_0",
                // 11.x
                "118_0", "117_0", "116_0", "115_0", "114_0", "113_0", "112_0", "111_0", "110_0",
                // 10.x
                "102_0", "101_0", "100_0"
            };

#ifdef _WIN32
            for (const auto& version : cudaVersions)
            {   //e.g. nvrtc64_128_0.dll
                std::string libraryName = executablePath + "\\nvrtc64_" + version + ".dll";
                DWORD fileAttr = GetFileAttributesA(libraryName.c_str());
                if (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY))  // file exists
                {
                    return libraryName;
                }
            }
#else
            std::string libraryName = executablePath + "/libnvrtc.so";
            if (access(libraryName.c_str(), F_OK) == 0)  // file exists
            {
                return libraryName;
            }
            for (const auto& version : cudaVersions)
            {   //e.g. libnvrtc.so.128_0
                std::string libraryName = executablePath + "/libnvrtc.so." + version;
                if (access(libraryName.c_str(), F_OK) == 0)  // file exists
                {
                    return libraryName;
                }
            }
#endif

            setLastError("No suitable NVRTC library found in the current executable directory: " + executablePath + ".");
            return "";
        }

        static bool initCudaDriverAPI()
        {
#ifdef _WIN32
            const std::string driver_dll = "nvcuda.dll";
#else
            const std::string driver_dll = "libcuda.so";
#endif
            auto* dlManager = whyb::DynamicLibraryManager::instance();
            auto driverLib = dlManager->loadLibrary(driver_dll);
            if (!driverLib)
            {
                setLastError("Failed to load NVIDIA Driver API library: " + driver_dll + ".");
                return false;
            }
            cuInit = (cuInit_t)(dlManager->getFunction(driver_dll, "cuInit"));
            cuDeviceGet = (cuDeviceGet_t)(dlManager->getFunction(driver_dll, "cuDeviceGet"));
            cuCtxCreate = (cuCtxCreate_t)(dlManager->getFunction(driver_dll, "cuCtxCreate_v2"));
            cuCtxDestroy = (cuCtxDestroy_t)(dlManager->getFunction(driver_dll, "cuCtxDestroy_v2"));
            cuStreamCreate = (cuStreamCreate_t)(dlManager->getFunction(driver_dll, "cuStreamCreate"));
            cuStreamDestroy = (cuStreamDestroy_t)(dlManager->getFunction(driver_dll, "cuStreamDestroy_v2"));
            cuStreamSynchronize = (cuStreamSynchronize_t)(dlManager->getFunction(driver_dll, "cuStreamSynchronize"));
            cuModuleLoadDataEx = (cuModuleLoadDataEx_t)(dlManager->getFunction(driver_dll, "cuModuleLoadDataEx"));
            cuModuleUnload = (cuModuleUnload_t)(dlManager->getFunction(driver_dll, "cuModuleUnload"));
            cuModuleGetFunction = (cuModuleGetFunction_t)(dlManager->getFunction(driver_dll, "cuModuleGetFunction"));
            cuLaunchKernel = (cuLaunchKernel_t)(dlManager->getFunction(driver_dll, "cuLaunchKernel"));
            cuCtxSynchronize = (cuCtxSynchronize_t)(dlManager->getFunction(driver_dll, "cuCtxSynchronize"));
            cuMemAlloc = (cuMemAlloc_t)(dlManager->getFunction(driver_dll, "cuMemAlloc_v2"));
            cuMemAllocHost = (cuMemAllocHost_t)(dlManager->getFunction(driver_dll, "cuMemAllocHost_v2"));
            cuMemFree = (cuMemFree_t)(dlManager->getFunction(driver_dll, "cuMemFree_v2"));
            cuMemFreeHost = (cuMemFreeHost_t)(dlManager->getFunction(driver_dll, "cuMemFreeHost"));
            cuMemcpyHtoD = (cuMemcpyHtoD_t)(dlManager->getFunction(driver_dll, "cuMemcpyHtoD_v2"));
            cuMemcpyDtoH = (cuMemcpyDtoH_t)(dlManager->getFunction(driver_dll, "cuMemcpyDtoH_v2"));

            if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuCtxDestroy ||
                !cuStreamCreate || !cuStreamDestroy || !cuStreamSynchronize ||
                !cuModuleLoadDataEx || !cuModuleUnload || !cuModuleGetFunction ||
                !cuLaunchKernel || !cuCtxSynchronize ||
                !cuMemAlloc || !cuMemAllocHost ||
                !cuMemFree || !cuMemFreeHost ||
                !cuMemcpyHtoD || !cuMemcpyDtoH) {
                setLastError("Failed to load one or more CUDA Driver API functions.");
                return false;
            }
            return true;
        }

        static bool initCudaFunctions(std::string& compiledPtxStr)
        {
            CUresult cuRes = cuInit(0);
            if (cuRes != 0) {
                setLastError("cuInit failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            CUdevice device;
            cuRes = cuDeviceGet(&device, 0);
            if (cuRes != 0) {
                setLastError("cuDeviceGet failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            cuRes = cuCtxCreate(&context, 0, device);
            if (cuRes != 0) {
                setLastError("cuCtxCreate failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            cuRes = cuStreamCreate(&cudastream, 0); //flag: CU_STREAM_DEFAULT = 0
            if (cuRes != 0) {
                setLastError("cuStreamCreate failed with error " + std::to_string(cuRes) + ".");
                return false;
            }

            // Load PTX module to GPU Memory
            cuRes = cuModuleLoadDataEx(&cudamodule, compiledPtxStr.c_str(), 0, nullptr, nullptr);
            if (cuRes != 0) {
                setLastError("cuModuleLoadDataEx failed with error " + std::to_string(cuRes) + ".");
                cuCtxDestroy(context);
                cuStreamDestroy(cudastream);
                return false;
            }

            // Get cuda module kernel function(cuda_hwc2chw)
            cuRes = cuModuleGetFunction(&hwc2chwCUDAFun, cudamodule, "cuda_hwc2chw");
            if (cuRes != 0) {
                setLastError("cuModuleGetFunction (cuda_hwc2chw) failed with error " + std::to_string(cuRes) + ".");
                cuModuleUnload(cudamodule);
                cuCtxDestroy(context);
                cuStreamDestroy(cudastream);
                return false;
            }
            // Get cuda module kernel function(cuda_chw2hwc)
            cuRes = cuModuleGetFunction(&chw2hwcCUDAFun, cudamodule, "cuda_chw2hwc");
            if (cuRes != 0) {
                setLastError("cuModuleGetFunction (cuda_chw2hwc) failed with error " + std::to_string(cuRes) + ".");
                cuModuleUnload(cudamodule);
                cuCtxDestroy(context);
                cuStreamDestroy(cudastream);
                return false;
            }
            return true;
        }

        static bool initAll()
        {
            std::lock_guard<std::mutex> lock(CUDAMutex);
            setLastError("");
            if (initCUDAStatus.load(std::memory_order_acquire) == InitCUDAStatusEnum::Ready) {
                std::string nvrtc_module_filename = findNVRTCModuleName();
                if (nvrtc_module_filename.empty()) {
                    setLastError("Could not find a suitable CUDA NVRTC library.");
                    lastCUDAErrorStr = "Could not found CUDA NVRTC dll failed.";
                    initCUDAStatus.store(InitCUDAStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                std::string ptx_str = compileCUDAWithNVRTC(nvrtc_module_filename, cudaSource);
                if (ptx_str.empty()) {
                    setLastError("Compile CUDA source code failed.");
                    lastCUDAErrorStr = "Compile CUDA Source code failed.";
                    initCUDAStatus.store(InitCUDAStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                bool init_cuda_driver = initCudaDriverAPI();
                if (!init_cuda_driver) {
                    setLastError("Failed to initialize the CUDA driver functions.");
                    lastCUDAErrorStr = "Failed to load CUDA Driver API functions.";
                    initCUDAStatus.store(InitCUDAStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                bool init_cuda_functions = initCudaFunctions(ptx_str);
                if (!init_cuda_functions) {
                    setLastError("Failed to initialize the CUDA driver functions.");
                    lastCUDAErrorStr = "Failed to load CUDA Driver API functions.";
                    initCUDAStatus.store(InitCUDAStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                initCUDAStatus.store(InitCUDAStatusEnum::Inited, std::memory_order_release);
                return true;
            }
            else if (initCUDAStatus.load(std::memory_order_acquire) == InitCUDAStatusEnum::Inited) {
                return true;
            }
            else if (initCUDAStatus.load(std::memory_order_acquire) == InitCUDAStatusEnum::Failed) {
                setLastError("CUDA initialization failed. " + lastError());
                return false;
            }
            return true;
        }

        static bool releaseAll()
        {
            std::lock_guard<std::mutex> lock(CUDAMutex);
            if (initCUDAStatus.load(std::memory_order_acquire) != InitCUDAStatusEnum::Inited)
            {
                return true;
            }

            CUresult cuRes = cuModuleUnload(cudamodule);
            if (cuRes != 0) {
                setLastError("cuModuleUnload failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            cuRes = cuStreamDestroy(cudastream);
            if (cuRes != 0) {
                setLastError("cuStreamDestroy failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            cuRes = cuCtxDestroy(context);
            if (cuRes != 0) {
                setLastError("cuCtxDestroy failed with error " + std::to_string(cuRes) + ".");
                return false;
            }
            auto* dlManager = whyb::DynamicLibraryManager::instance();
#ifdef _WIN32
            const std::string driver_dll = "nvcuda.dll";
#else
            const std::string driver_dll = "libcuda.so";
#endif
            dlManager->unloadLibrary(driver_dll);
            consecutiveCUDAFailures.store(0, std::memory_order_release);
            initCUDAStatus.store(InitCUDAStatusEnum::Ready, std::memory_order_release);
            return true;
        }
    private:
        inline static std::atomic<InitCUDAStatusEnum> initCUDAStatus = InitCUDAStatusEnum::Ready;
        inline static std::string lastCUDAErrorStr = "";
        inline static std::mutex CUDAMutex;
        inline static std::mutex errorMutex;
        inline static std::atomic<size_t> consecutiveCUDAFailures = 0;
        static constexpr size_t maxConsecutiveCUDAFailures = 3;

        inline static CUfunction hwc2chwCUDAFun = nullptr;
        inline static CUfunction chw2hwcCUDAFun = nullptr;
        inline static CUcontext context;
        inline static CUstream cudastream = nullptr;
        inline static CUmodule cudamodule = nullptr;
    };

} // namespace whyb
