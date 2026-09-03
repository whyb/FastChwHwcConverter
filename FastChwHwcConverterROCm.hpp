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

// HIPRTC enum type define
typedef enum {
    HIPRTC_SUCCESS = 0,
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    HIPRTC_ERROR_INVALID_INPUT = 3,
    HIPRTC_ERROR_INVALID_PROGRAM = 4,
    HIPRTC_ERROR_INVALID_OPTION = 5,
    HIPRTC_ERROR_COMPILATION = 6,
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    HIPRTC_ERROR_INTERNAL_ERROR = 11
} hiprtcResult;
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,            ///< Host-to-Host Copy
    hipMemcpyHostToDevice = 1,          ///< Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,          ///< Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3,        ///< Device-to-Device Copy
    hipMemcpyDefault = 4,               ///< Runtime will automatically determine
                                        ///<copy-kind based on virtual addresses.
    hipMemcpyDeviceToDeviceNoCU = 1024  ///< Device-to-Device Copy without using compute units
} hipMemcpyKind;
typedef struct _hiprtcProgram* hiprtcProgram;

// HIPRTC function type define
typedef hiprtcResult(*hiprtcCreateProgram_t)(hiprtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef hiprtcResult(*hiprtcCompileProgram_t)(hiprtcProgram, int, const char* const*);
typedef hiprtcResult(*hiprtcGetCodeSize_t)(hiprtcProgram, size_t*);
typedef hiprtcResult(*hiprtcGetCode_t)(hiprtcProgram, char*);
typedef hiprtcResult(*hiprtcDestroyProgram_t)(hiprtcProgram*);
typedef hiprtcResult(*hiprtcGetProgramLogSize_t)(hiprtcProgram, size_t*);
typedef hiprtcResult(*hiprtcGetProgramLog_t)(hiprtcProgram, char*);
typedef const char* (*hiprtcGetErrorString_t)(hiprtcResult);

// AMD ROCm Driver API data type define
typedef int hipError_t;
typedef int hipDevice_t;
typedef void* hipCtx_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef unsigned long long hipDeviceptr_t;
typedef void* hipStream_t;
typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z) {};
} dim3;
typedef struct ihipEvent_t* hipEvent_t;

// AMD ROCm Driver API function type define
typedef hipError_t(*hipInit_t)(unsigned int);
typedef hipError_t(*hipDeviceGet_t)(hipDevice_t*, int);
typedef hipError_t(*hipCtxCreate_t)(hipCtx_t*, unsigned int, hipDevice_t);
typedef hipError_t(*hipCtxDestroy_t)(hipCtx_t);
typedef hipError_t(*hipStreamCreate_t)(hipStream_t*);
typedef hipError_t(*hipStreamDestroy_t)(hipStream_t);
typedef hipError_t(*hipStreamSynchronize_t)(hipStream_t);
typedef hipError_t(*hipModuleLoadDataEx_t)(hipModule_t*, const void*, unsigned int, int*, void**);
typedef hipError_t(*hipModuleUnload_t)(hipModule_t);
typedef hipError_t(*hipModuleGetFunction_t)(hipFunction_t*, hipModule_t, const char*);
typedef hipError_t(*hipLaunchKernel_t)(hipFunction_t, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t(*hipModuleLaunchKernel_t)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**);
typedef hipError_t(*hipCtxSynchronize_t)(void);

typedef hipError_t(*hipMalloc_t)(hipDeviceptr_t*, size_t);
typedef hipError_t(*hipMallocAsync_t)(hipDeviceptr_t*, size_t, hipStream_t);
typedef hipError_t(*hipHostMalloc_t)(void**, size_t, unsigned int);
typedef hipError_t(*hipFree_t)(hipDeviceptr_t);
typedef hipError_t(*hipFreeAsync_t)(hipDeviceptr_t, hipStream_t);
typedef hipError_t(*hipHostFree_t)(void*);

typedef hipError_t(*hipMemcpy_t)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t(*hipMemcpyAsync_t)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
typedef hipError_t(*hipMemcpyHtoD_t)(hipDeviceptr_t, const void*, size_t);
typedef hipError_t(*hipMemcpyHtoDAsync_t)(hipDeviceptr_t, const void*, size_t, hipStream_t);
typedef hipError_t(*hipMemcpyDtoH_t)(void*, hipDeviceptr_t, size_t);
typedef hipError_t(*hipMemcpyDtoHAsync_t)(void*, hipDeviceptr_t, size_t, hipStream_t);

typedef hipError_t(*hipEventCreate_t)(hipEvent_t*);
typedef hipError_t(*hipEventRecord_t)(hipEvent_t, hipStream_t);
typedef hipError_t(*hipEventSynchronize_t)(hipEvent_t);
typedef hipError_t(*hipEventElapsedTime_t)(float*, hipEvent_t, hipEvent_t);
typedef hipError_t(*hipEventDestroy_t)(hipEvent_t);

#ifdef _WIN32
#define DYNAMIC_LIBRARY_EXTENSION ".dll"
#else
#define DYNAMIC_LIBRARY_EXTENSION ".so"
#endif

// ROCm Driver API function pointers. These are inline variables (one shared
// object across translation units) instead of file-scope statics, so every TU
// that includes this header observes the same initialized set.
inline hipInit_t hipInit = nullptr;
inline hipDeviceGet_t hipDeviceGet = nullptr;
inline hipCtxCreate_t hipCtxCreate = nullptr;
inline hipCtxDestroy_t hipCtxDestroy = nullptr;
inline hipStreamCreate_t hipStreamCreate = nullptr;
inline hipStreamDestroy_t hipStreamDestroy = nullptr;
inline hipStreamSynchronize_t hipStreamSynchronize = nullptr;
inline hipModuleLoadDataEx_t hipModuleLoadDataEx = nullptr;
inline hipModuleUnload_t hipModuleUnload = nullptr;
inline hipModuleGetFunction_t hipModuleGetFunction = nullptr;
inline hipLaunchKernel_t hipLaunchKernel = nullptr;
inline hipModuleLaunchKernel_t hipModuleLaunchKernel = nullptr;
inline hipCtxSynchronize_t hipCtxSynchronize = nullptr;
inline hipMalloc_t hipMalloc = nullptr;
inline hipMallocAsync_t hipMallocAsync = nullptr;
inline hipHostMalloc_t hipHostMalloc = nullptr;
inline hipFree_t hipFree = nullptr;
inline hipFreeAsync_t hipFreeAsync = nullptr;
inline hipHostFree_t hipHostFree = nullptr;
inline hipMemcpy_t hipMemcpy = nullptr;
inline hipMemcpyAsync_t hipMemcpyAsync = nullptr;
inline hipMemcpyHtoD_t hipMemcpyHtoD = nullptr;
inline hipMemcpyHtoDAsync_t hipMemcpyHtoDAsync = nullptr;
inline hipMemcpyDtoH_t hipMemcpyDtoH = nullptr;
inline hipMemcpyDtoHAsync_t hipMemcpyDtoHAsync = nullptr;
inline hipEventCreate_t hipEventCreate = nullptr;
inline hipEventRecord_t hipEventRecord = nullptr;
inline hipEventSynchronize_t hipEventSynchronize = nullptr;
inline hipEventElapsedTime_t hipEventElapsedTime = nullptr;
inline hipEventDestroy_t hipEventDestroy = nullptr;

static const char* rocmSource = R"(
typedef unsigned char uint8_t;

// HWC -> CHW
extern "C" __global__ void rocm_hwc2chw(const size_t h, const size_t w, const size_t c,
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
extern "C" __global__ void rocm_chw2hwc(const size_t c, const size_t h, const size_t w,
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

enum struct InitROCmStatusEnum : int
    {
        Ready = 0,
        Inited = 1,
        Failed = 2,
    };
    class amd {
    private:
        amd() {
            static bool init0([]() {
                initAll();
                return true; 
            }());
        }
    public:
        ~amd() = default;
        amd(const amd&) = delete;
        amd& operator=(const amd&) = delete;
        amd(amd&&) = delete;
        amd& operator=(amd&&) = delete;
    public:
        static bool init() { return initAll(); }
        static bool release() { return releaseAll(); }

        // Query initialization state and the last backend error without
        // requiring library-side terminal diagnostics.
        static InitROCmStatusEnum status() { return initROCmStatus.load(std::memory_order_acquire); }
        static std::string lastError()
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            return lastROCmErrorStr;
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
            amd();
            if (initROCmStatus.load(std::memory_order_acquire) != InitROCmStatusEnum::Inited) {
                // use cpu
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha); return;
            }
            // Initialize sizes
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(uint8_t);
            const size_t output_size = pixel_size * sizeof(float);

            hipDeviceptr_t rocm_input_memory = 0;
            hipDeviceptr_t rocm_output_memory = 0;

            // Allocate host-pinned memory
            hipError_t hipRes0 = hipMallocAsync(&rocm_input_memory, input_size, rocmstream);
            hipError_t hipRes1 = hipMallocAsync(&rocm_output_memory, output_size, rocmstream);

            if (hipRes0 != 0 || hipRes1 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "ROCm hwc2chw GPU execution failed");
                return;
            }

            // Copy host memory to device memory
            hipError_t hipRes2 = hipMemcpyHtoDAsync(rocm_input_memory, src, input_size, rocmstream);

            if (hipRes2 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "ROCm hwc2chw GPU execution failed");
                return;
            }

            // Kernel execution
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;

            size_t arg_h_val = h, arg_w_val = w, arg_c_val = c;
            float arg_alpha_val = alpha;
            void* args[] = { &arg_h_val, &arg_w_val, &arg_c_val, &rocm_input_memory, &rocm_output_memory, &arg_alpha_val };

            hipError_t hipRes3 = hipModuleLaunchKernel(
                hwc2chwROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);

            if (hipRes3 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "ROCm hwc2chw GPU execution failed");
                return;
            }

            // Copy device memory to host memory
            hipError_t hipRes5 = hipMemcpyDtoHAsync(dst, rocm_output_memory, output_size, rocmstream);

            if (hipRes5 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuHwc2chw(h, w, c, src, dst, alpha, "ROCm hwc2chw GPU execution failed");
                return;
            }

            // Free memory
            hipFreeAsync(rocm_input_memory, rocmstream);
            hipFreeAsync(rocm_output_memory, rocmstream);

            // Stream synchronization
            hipError_t hipRes4 = hipStreamSynchronize(rocmstream);

            if (hipRes4 != 0) {
                recordCallFailure("ROCm hwc2chw synchronization failed");
                return;
            }
            clearCallFailures();
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
            amd();
            if (initROCmStatus.load(std::memory_order_acquire) != InitROCmStatusEnum::Inited) {
                // use cpu
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha); return;
            }
            // use rocm
            const size_t pixel_size = h * w * c;
            size_t input_size = pixel_size * sizeof(float);
            size_t output_size = pixel_size * sizeof(uint8_t);
            hipDeviceptr_t rocm_input_memory = 0;
            hipDeviceptr_t rocm_output_memory = 0;

            // Allocate device memory
            hipError_t hipRes0 = hipMallocAsync(&rocm_input_memory, input_size, rocmstream);
            hipError_t hipRes1 = hipMallocAsync(&rocm_output_memory, output_size, rocmstream);

            if (hipRes0 != 0 || hipRes1 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "ROCm chw2hwc GPU execution failed"); return;
            }

            // Copy host memory to device memory
            hipError_t hipRes2 = hipMemcpyHtoDAsync(rocm_input_memory, src, input_size, rocmstream);

            if (hipRes2 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "ROCm chw2hwc GPU execution failed"); return;
            }

            // Call kernel
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;

            size_t arg_c_val = c;
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            uint8_t arg_alpha_val = alpha;
            void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &rocm_input_memory, &rocm_output_memory, &arg_alpha_val };

            hipError_t hipRes3 = hipModuleLaunchKernel(
                chw2hwcROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);

            if (hipRes3 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "ROCm chw2hwc GPU execution failed"); return;
            }

            // Copy device memory to host memory
            hipError_t hipRes5 = hipMemcpyDtoHAsync(dst, rocm_output_memory, output_size, rocmstream);

            if (hipRes5 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                fallbackToCpuChw2hwc(c, h, w, src, dst, alpha, "ROCm chw2hwc GPU execution failed"); return;
            }

            // Free memory
            hipFreeAsync(rocm_input_memory, rocmstream);
            hipFreeAsync(rocm_output_memory, rocmstream);
            hipError_t hipRes4 = hipStreamSynchronize(rocmstream);

            if (hipRes4 != 0) {

                recordCallFailure("ROCm chw2hwc synchronization failed");
                return;
            }
            clearCallFailures();
        }

        /**
         * @brief Converts image data from HWC format to CHW format
         *
         * @param h Height of image
         * @param w Width of image
         * @param c Number of channels
         * @param src ROCm Memory (uint8_t) Pointer to the source data in HWC format
         * @param dst ROCm Memory (float) Pointer to the destination data in CHW format
         * @param alpha Scaling factor
         */
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            hipDeviceptr_t src, hipDeviceptr_t dst,
            const float alpha = 1.f / 255.f) {
            amd();
            if (initROCmStatus.load(std::memory_order_acquire) != InitROCmStatusEnum::Inited) {
                setLastError("ROCm device-memory hwc2chw called before successful initialization.");
                return;
            }
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(uint8_t);
            const size_t output_size = pixel_size * sizeof(float);

            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready rocm kernel function(func_hwc2chw)
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            size_t arg_c_val = c;
            float arg_alpha_val = alpha;
            void* args[] = { &arg_h_val, &arg_w_val, &arg_c_val, &src, &dst, &arg_alpha_val };
            hipError_t hipRes0 = hipModuleLaunchKernel(
                hwc2chwROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);
            if (hipRes0 != 0) {
                recordCallFailure("ROCm device-memory hwc2chw launch failed");
                return;
            }
            hipError_t hipRes1 = hipStreamSynchronize(rocmstream);
            if (hipRes1 != 0) {
                recordCallFailure("ROCm device-memory hwc2chw synchronization failed");
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
         * @param src ROCm Memory (float) Pointer to the source data in CHW format
         * @param dst ROCm Memory (uint8_t) Pointer to the destination data in HWC format
         * @param alpha Scaling factor
         */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            hipDeviceptr_t src, hipDeviceptr_t dst,
            const uint8_t alpha = 255.0f) {
            amd();
            if (initROCmStatus.load(std::memory_order_acquire) != InitROCmStatusEnum::Inited) {
                setLastError("ROCm device-memory chw2hwc called before successful initialization.");
                return;
            }
            const unsigned int blockDimX = 32, blockDimY = 32, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready rocm kernel function(func_hwc2chw)
            size_t arg_c_val = c;
            size_t arg_h_val = h;
            size_t arg_w_val = w;
            uint8_t arg_alpha_val = alpha;
            void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &src, &dst, &arg_alpha_val };
            hipError_t hipRes0 = hipModuleLaunchKernel(
                chw2hwcROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);
            if (hipRes0 != 0) {
                recordCallFailure("ROCm device-memory chw2hwc launch failed");
                return;
            }
            hipError_t hipRes1 = hipStreamSynchronize(rocmstream);
            if (hipRes1 != 0) {
                recordCallFailure("ROCm device-memory chw2hwc synchronization failed");
                return;
            }
            clearCallFailures();
            return;
        }

    private:

        static bool isFileExists(const std::string& path)
        {
        #ifdef _WIN32
            DWORD fileAttr = GetFileAttributesA(path.c_str());
            return (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY));
        #else
            return (access(path.c_str(), F_OK) == 0);
        #endif
        }

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
            lastROCmErrorStr = message;
        }

        // Repeated transient failures stop repeated expensive GPU attempts.
        // The backend remains available after release()/init() resets state.
        static void recordCallFailure(const std::string& message)
        {
            const auto failures = ++consecutiveROCmFailures;
            const std::string fullMessage = message + " (consecutive ROCm failures: " +
                std::to_string(failures) + ").";
            setLastError(fullMessage);

            if (failures >= maxConsecutiveROCmFailures)
            {
                std::lock_guard<std::mutex> lock(ROCmMutex);
                initROCmStatus.store(InitROCmStatusEnum::Failed, std::memory_order_release);
                setLastError(fullMessage + " ROCm backend disabled until release()/init().");
            }
        }

        static void clearCallFailures()
        {
            consecutiveROCmFailures.store(0, std::memory_order_release);
        }

        static std::string compileROCmWithHIPRTC(const std::string& libraryName, const std::string& rocmSource)
        {
            // dynamic load HIPRTC lib
            auto* dlManager = DynamicLibraryManager::instance();
            auto hiprtcLib = dlManager->loadLibrary(libraryName);
            if (!hiprtcLib)
            {
                setLastError("Failed to load HIPRTC library: " + libraryName + ".");
                return "";
            }

            // Get HIPRTC function points
            auto hiprtcCreateProgram_fun = (hiprtcCreateProgram_t)(dlManager->getFunction(libraryName, "hiprtcCreateProgram"));
            auto hiprtcCompileProgram_fun = (hiprtcCompileProgram_t)(dlManager->getFunction(libraryName, "hiprtcCompileProgram"));
            auto hiprtcGetCodeSize_fun = (hiprtcGetCodeSize_t)(dlManager->getFunction(libraryName, "hiprtcGetCodeSize")); // hiprtcGetBitcodeSize or hiprtcGetCodeSize
            auto hiprtcGetCode_fun = (hiprtcGetCode_t)(dlManager->getFunction(libraryName, "hiprtcGetCode")); // hiprtcGetBitcode or hiprtcGetCode
            auto hiprtcDestroyProgram_fun = (hiprtcDestroyProgram_t)(dlManager->getFunction(libraryName, "hiprtcDestroyProgram"));
            auto hiprtcGetProgramLogSize_fun = (hiprtcGetProgramLogSize_t)(dlManager->getFunction(libraryName, "hiprtcGetProgramLogSize"));
            auto hiprtcGetProgramLog_fun = (hiprtcGetProgramLog_t)(dlManager->getFunction(libraryName, "hiprtcGetProgramLog"));
            auto hiprtcGetErrorString_fun = (hiprtcGetErrorString_t)(dlManager->getFunction(libraryName, "hiprtcGetErrorString"));

            // Check function point is not nullptr
            if (!hiprtcCreateProgram_fun || !hiprtcCompileProgram_fun || !hiprtcGetCodeSize_fun ||
                !hiprtcGetCode_fun || !hiprtcDestroyProgram_fun || !hiprtcGetProgramLogSize_fun ||
                !hiprtcGetProgramLog_fun || !hiprtcGetErrorString_fun)
            {
                setLastError("Failed to load HIPRTC functions from: " + libraryName + ".");
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // Create HIPRTC Program Object
            hiprtcProgram prog;
            const char* headers[] = { 0 };

            const char* includeNames[] = { 0 };
            hiprtcResult res = hiprtcCreateProgram_fun(&prog, rocmSource.c_str(), "FastChwHwcConverterROCm.cu", 0, headers, includeNames);
            if (res != HIPRTC_SUCCESS)
            {
                setLastError(std::string("hiprtcCreateProgram failed: ") + hiprtcGetErrorString_fun(res));
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // compile ROCm source code
            const char* options[] = { "-default-device", "--std=c++11" };
            res = hiprtcCompileProgram_fun(prog, 2, options);
            if (res != HIPRTC_SUCCESS)
            {
                size_t logSize;
                hiprtcGetProgramLogSize_fun(prog, &logSize);
                std::string log(logSize, '\0');
                hiprtcGetProgramLog_fun(prog, &log[0]);
                setLastError("ROCm Compile error: " + log);
                hiprtcDestroyProgram_fun(&prog);
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // Get Code String
            size_t codeSize;
            hiprtcGetCodeSize_fun(prog, &codeSize);
            std::string code(codeSize, '\0');
            hiprtcGetCode_fun(prog, &code[0]);

            // Release HIPRTC Program
            hiprtcDestroyProgram_fun(&prog);

            // Release library
            dlManager->unloadLibrary(libraryName);
            return code;
        }

        static std::string findHIPRTCModuleName()
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
            // ROCm version list: v5.0 ~ v7.2.
            // ref: https://rocm.docs.amd.com/en/latest/release/versions.html
            const std::vector<std::string> rocmVersions = {
                "0702", "0701", "0700", // driver: amdhip64_7.dll
                "0604", "0603", "0602", "0601", "0600", // driver: amdhip64_6.dll
                "0507", "0506", "0505", "0504", "0502", "0501", "0500"  // driver: amdhip64.dll
            };

#ifdef _WIN32
            for (const auto& version : rocmVersions)
            {   //e.g. hiprtc0602.dll
                std::string libraryPath = executablePath + "\\hiprtc" + version + ".dll";
                if (isFileExists(libraryPath))
                {
                    return libraryPath;
                }
            }
#else
            std::string libraryName = executablePath + "/libhiprtc.so";
            if (access(libraryName.c_str(), F_OK) == 0)  // file exists
            {
                return libraryName;
            }
            for (const auto& version : rocmVersions)
            {   //e.g. libhiprtc.so.0602
                std::string libraryPath = executablePath + "/libhiprtc.so." + version;
                if (isFileExists(libraryPath))
                {
                    return libraryPath;
                }
            }
#endif

            setLastError("No suitable HIPRTC library found in the current executable directory: " + executablePath + ".");
            return "";
        }

        static bool initROCmDriverAPI()
        {
#ifdef _WIN32
            const std::vector<std::string> candidates = {
                "amdhip64_7.dll",
                "amdhip64_6.dll",
                "amdhip64.dll"
            };
#else
            const std::vector<std::string> candidates = {
                "amdhip64.so",
            };
#endif
            auto* dlManager = whyb::DynamicLibraryManager::instance();
            std::string driver_dll = "";
            void* driverLib = nullptr;

            for (const auto& name : candidates) {
                driverLib = dlManager->loadLibrary(name);
                if (driverLib) {
                    driver_dll = name;
                    break;
                }
            }

            if (!driverLib) {
                setLastError("Failed to load any AMD ROCm driver library (tried v7, v6, v5).");
                return false;
            }

            hipInit = (hipInit_t)(dlManager->getFunction(driver_dll, "hipInit"));
            hipDeviceGet = (hipDeviceGet_t)(dlManager->getFunction(driver_dll, "hipDeviceGet"));
            hipCtxCreate = (hipCtxCreate_t)(dlManager->getFunction(driver_dll, "hipCtxCreate"));
            hipCtxDestroy = (hipCtxDestroy_t)(dlManager->getFunction(driver_dll, "hipCtxDestroy"));
            hipStreamCreate = (hipStreamCreate_t)(dlManager->getFunction(driver_dll, "hipStreamCreate"));
            hipStreamDestroy = (hipStreamDestroy_t)(dlManager->getFunction(driver_dll, "hipStreamDestroy"));
            hipStreamSynchronize = (hipStreamSynchronize_t)(dlManager->getFunction(driver_dll, "hipStreamSynchronize"));
            hipModuleLoadDataEx = (hipModuleLoadDataEx_t)(dlManager->getFunction(driver_dll, "hipModuleLoadDataEx"));
            hipModuleUnload = (hipModuleUnload_t)(dlManager->getFunction(driver_dll, "hipModuleUnload"));
            hipModuleGetFunction = (hipModuleGetFunction_t)(dlManager->getFunction(driver_dll, "hipModuleGetFunction"));
            hipLaunchKernel = (hipLaunchKernel_t)(dlManager->getFunction(driver_dll, "hipLaunchKernel"));
            hipModuleLaunchKernel = (hipModuleLaunchKernel_t)(dlManager->getFunction(driver_dll, "hipModuleLaunchKernel"));
            hipCtxSynchronize = (hipCtxSynchronize_t)(dlManager->getFunction(driver_dll, "hipCtxSynchronize"));
            //hipMemAlloc = (hipMemAlloc_t)(dlManager->getFunction(driver_dll, "hipMemAlloc"));
            hipMalloc = (hipMalloc_t)(dlManager->getFunction(driver_dll, "hipMalloc")); // == hipMemAlloc
            hipMallocAsync = (hipMallocAsync_t)(dlManager->getFunction(driver_dll, "hipMallocAsync"));
            hipHostMalloc = (hipHostMalloc_t)(dlManager->getFunction(driver_dll, "hipHostMalloc"));
            //hipMemFree = (hipMemFree_t)(dlManager->getFunction(driver_dll, "hipMemFree"));
            hipFree = (hipFree_t)(dlManager->getFunction(driver_dll, "hipFree")); // == hipMemFree
            hipFreeAsync = (hipFreeAsync_t)(dlManager->getFunction(driver_dll, "hipFreeAsync"));
            hipHostFree = (hipHostFree_t)(dlManager->getFunction(driver_dll, "hipHostFree"));
            hipMemcpy = (hipMemcpy_t)(dlManager->getFunction(driver_dll, "hipMemcpy"));
            hipMemcpyAsync = (hipMemcpyAsync_t)(dlManager->getFunction(driver_dll, "hipMemcpyAsync"));
            hipMemcpyHtoD = (hipMemcpyHtoD_t)(dlManager->getFunction(driver_dll, "hipMemcpyHtoD"));
            hipMemcpyHtoDAsync = (hipMemcpyHtoDAsync_t)(dlManager->getFunction(driver_dll, "hipMemcpyHtoDAsync"));
            hipMemcpyDtoH = (hipMemcpyDtoH_t)(dlManager->getFunction(driver_dll, "hipMemcpyDtoH"));
            hipMemcpyDtoHAsync = (hipMemcpyDtoHAsync_t)(dlManager->getFunction(driver_dll, "hipMemcpyDtoHAsync"));
            hipEventCreate = (hipEventCreate_t)(dlManager->getFunction(driver_dll, "hipEventCreate"));
            hipEventRecord = (hipEventRecord_t)(dlManager->getFunction(driver_dll, "hipEventRecord"));
            hipEventSynchronize = (hipEventSynchronize_t)(dlManager->getFunction(driver_dll, "hipEventSynchronize"));
            hipEventElapsedTime = (hipEventElapsedTime_t)(dlManager->getFunction(driver_dll, "hipEventElapsedTime"));
            hipEventDestroy = (hipEventDestroy_t)(dlManager->getFunction(driver_dll, "hipEventDestroy"));

            if (!hipInit || !hipDeviceGet || !hipCtxCreate ||
                !hipStreamCreate || !hipStreamDestroy || !hipStreamSynchronize ||
                !hipModuleLoadDataEx ||
                !hipModuleGetFunction || !hipLaunchKernel || !hipCtxSynchronize ||
                !hipMalloc || !hipMallocAsync || !hipHostMalloc || 
                !hipFree || !hipFreeAsync || !hipHostFree ||
                !hipMemcpy || !hipMemcpyAsync ||
                !hipMemcpyHtoD || !hipMemcpyHtoDAsync ||
                !hipMemcpyDtoH || !hipMemcpyDtoHAsync ||
                !hipEventCreate || !hipEventRecord ||
                !hipEventSynchronize || !hipEventElapsedTime || !hipEventDestroy
                ) {
                setLastError("Failed to load one or more AMD ROCm driver functions.");
                return false;
            }
            return true;
        }

        static bool initROCmFunctions(std::string& compiledPtxStr)
        {
            hipError_t hipRes = hipInit(0);
            if (hipRes != 0) {
                setLastError("hipInit failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            hipDevice_t device;
            hipRes = hipDeviceGet(&device, 0);
            if (hipRes != 0) {
                setLastError("hipDeviceGet failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            hipRes = hipCtxCreate(&context, 0, device);
            if (hipRes != 0) {
                setLastError("hipCtxCreate failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            hipRes = hipStreamCreate(&rocmstream);
            if (hipRes != 0) {
                setLastError("hipStreamCreate failed with error " + std::to_string(hipRes) + ".");
                return false;
            }

            // Load Code(like PTX) module to GPU memory
            hipRes = hipModuleLoadDataEx(&rocmmodule, compiledPtxStr.c_str(), 0, nullptr, nullptr);
            if (hipRes != 0) {
                setLastError("hipModuleLoadDataEx failed with error " + std::to_string(hipRes) + ".");
                hipCtxDestroy(context);
                hipStreamDestroy(rocmstream);
                return false;
            }

            // Get ROCm module kernel function(rocm_hwc2chw)
            hipRes = hipModuleGetFunction(&hwc2chwROCmFun, rocmmodule, "rocm_hwc2chw");
            if (hipRes != 0) {
                setLastError("hipModuleGetFunction (rocm_hwc2chw) failed with error " + std::to_string(hipRes) + ".");
                hipModuleUnload(rocmmodule);
                hipCtxDestroy(context);
                hipStreamDestroy(rocmstream);
                return false;
            }
            // Get ROCm module kernel function(rocm_chw2hwc)
            hipRes = hipModuleGetFunction(&chw2hwcROCmFun, rocmmodule, "rocm_chw2hwc");
            if (hipRes != 0) {
                setLastError("hipModuleGetFunction (rocm_chw2hwc) failed with error " + std::to_string(hipRes) + ".");
                hipModuleUnload(rocmmodule);
                hipCtxDestroy(context);
                hipStreamDestroy(rocmstream);
                return false;
            }
            return true;
        }

        static bool initAll()
        {
            std::lock_guard<std::mutex> lock(ROCmMutex);
            setLastError("");
            if (initROCmStatus.load(std::memory_order_acquire) == InitROCmStatusEnum::Ready) {
                std::string hiprtc_module_filename = findHIPRTCModuleName();
                if (hiprtc_module_filename.empty()) {
                    setLastError("Could not find a suitable AMD ROCm HIPRTC library.");
                    lastROCmErrorStr = "Could not found AMD ROCm HIPRTC dll failed.";
                    initROCmStatus.store(InitROCmStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                std::string code_str = compileROCmWithHIPRTC(hiprtc_module_filename, rocmSource);
                if (code_str.empty()) {
                    setLastError("Compile ROCm source code failed.");
                    lastROCmErrorStr = "Compile ROCm Source code failed.";
                    initROCmStatus.store(InitROCmStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                bool init_rocm_driver = initROCmDriverAPI();
                if (!init_rocm_driver) {
                    setLastError("Failed to initialize the ROCm driver functions.");
                    lastROCmErrorStr = "Failed to load ROCm Driver API functions.";
                    initROCmStatus.store(InitROCmStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                bool init_rocm_functions = initROCmFunctions(code_str);
                if (!init_rocm_functions) {
                    setLastError("Failed to initialize the ROCm driver functions.");
                    lastROCmErrorStr = "Failed to load ROCm Driver API functions.";
                    initROCmStatus.store(InitROCmStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                initROCmStatus.store(InitROCmStatusEnum::Inited, std::memory_order_release);
                return true;
            }
            else if (initROCmStatus.load(std::memory_order_acquire) == InitROCmStatusEnum::Inited) {
                return true;
            }
            else if (initROCmStatus.load(std::memory_order_acquire) == InitROCmStatusEnum::Failed) {
                setLastError("ROCm initialization failed. " + lastError());
                return false;
            }
            return true;
        }

        static bool releaseAll()
        {
            std::lock_guard<std::mutex> lock(ROCmMutex);
            if (initROCmStatus.load(std::memory_order_acquire) != InitROCmStatusEnum::Inited)
            {
                return true;
            }

            hipError_t hipRes = hipModuleUnload(rocmmodule);
            if (hipRes != 0) {
                setLastError("hipModuleUnload failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            hipRes = hipStreamDestroy(rocmstream);
            if (hipRes != 0) {
                setLastError("hipStreamDestroy failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            hipRes = hipCtxDestroy(context);
            if (hipRes != 0) {
                setLastError("hipCtxDestroy failed with error " + std::to_string(hipRes) + ".");
                return false;
            }
            auto* dlManager = whyb::DynamicLibraryManager::instance();
#ifdef _WIN32
            const std::string driver_dll = "amdhip64_6.dll"; // ROCm v6.x: amdhip64_6.dll, ROCm v5.x: amdhip64.dll
#else
            const std::string driver_dll = "amdhip64_6.so";
#endif
            dlManager->unloadLibrary(driver_dll);
            consecutiveROCmFailures.store(0, std::memory_order_release);
            initROCmStatus.store(InitROCmStatusEnum::Ready, std::memory_order_release);
            return true;
        }

    private:
        inline static std::atomic<InitROCmStatusEnum> initROCmStatus = InitROCmStatusEnum::Ready;
        inline static std::string lastROCmErrorStr = "";
        inline static std::mutex ROCmMutex;
        inline static std::mutex errorMutex;
        inline static std::atomic<size_t> consecutiveROCmFailures = 0;
        static constexpr size_t maxConsecutiveROCmFailures = 3;

        inline static hipFunction_t hwc2chwROCmFun = nullptr;
        inline static hipFunction_t chw2hwcROCmFun = nullptr;
        inline static hipCtx_t context;
        inline static hipStream_t rocmstream = nullptr;
        inline static hipModule_t rocmmodule = nullptr;
    };

} // namespace whyb
