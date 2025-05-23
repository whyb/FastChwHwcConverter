/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2025] [張小凡](https://github.com/whyb)
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

#include <iostream>
#include <vector>
#include <string>
#include <mutex>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

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

// static ROCm Driver API function points
static hipInit_t hipInit = nullptr;
static hipDeviceGet_t hipDeviceGet = nullptr;
static hipCtxCreate_t hipCtxCreate = nullptr;
static hipCtxDestroy_t hipCtxDestroy = nullptr;
static hipStreamCreate_t hipStreamCreate = nullptr;
static hipStreamDestroy_t hipStreamDestroy = nullptr;
static hipStreamSynchronize_t hipStreamSynchronize = nullptr;
static hipModuleLoadDataEx_t hipModuleLoadDataEx = nullptr;
static hipModuleUnload_t hipModuleUnload = nullptr;
static hipModuleGetFunction_t hipModuleGetFunction = nullptr;
static hipLaunchKernel_t hipLaunchKernel = nullptr;
static hipModuleLaunchKernel_t hipModuleLaunchKernel = nullptr;
static hipCtxSynchronize_t hipCtxSynchronize = nullptr;
static hipMalloc_t hipMalloc = nullptr;
static hipMallocAsync_t hipMallocAsync = nullptr;
static hipHostMalloc_t hipHostMalloc = nullptr;
static hipFree_t hipFree = nullptr;
static hipFreeAsync_t hipFreeAsync = nullptr;
static hipHostFree_t hipHostFree = nullptr;
static hipMemcpy_t hipMemcpy = nullptr;
static hipMemcpyAsync_t hipMemcpyAsync = nullptr;
static hipMemcpyHtoD_t hipMemcpyHtoD = nullptr;
static hipMemcpyHtoDAsync_t hipMemcpyHtoDAsync = nullptr;
static hipMemcpyDtoH_t hipMemcpyDtoH = nullptr;
static hipMemcpyDtoHAsync_t hipMemcpyDtoHAsync = nullptr;
static hipEventCreate_t hipEventCreate = nullptr;
static hipEventRecord_t hipEventRecord = nullptr;
static hipEventSynchronize_t hipEventSynchronize = nullptr;
static hipEventElapsedTime_t hipEventElapsedTime = nullptr;
static hipEventDestroy_t hipEventDestroy = nullptr;


static const char* rocmSource = R"(
typedef unsigned char uint8_t;

// HWC -> CHW
extern "C" __global__ void rocm_hwc2chw(const size_t h, const size_t w, const size_t c,
                                        const uint8_t* __restrict__ src, float* __restrict__ dst, const float alpha = 1.0f) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int dz = blockIdx.z * blockDim.z + threadIdx.z;
  
    if (dx < w && dy < h && dz < c) {
        size_t src_idx = dy * w * c + dx * c + dz;
        size_t dst_idx = dz * w * h + dy * w + dx;
        dst[dst_idx] = static_cast<float>(src[src_idx] * alpha);
    }
}

// CHW -> HWC
extern "C" __global__ void rocm_chw2hwc(const size_t c, const size_t h, const size_t w,
                                        const float* __restrict__ src, uint8_t* __restrict__ dst, const uint8_t alpha = 1) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int dz = blockIdx.z * blockDim.z + threadIdx.z;
  
    if (dx < w && dy < h && dz < c) {
        size_t src_idx = dz * w * h + dy * w + dx;
        size_t dst_idx = dy * w * c + dx * c + dz;
        dst[dst_idx] = static_cast<uint8_t>(src[src_idx] * alpha);
    }
}

)";

namespace whyb {
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
            if (initROCmStatus != InitROCmStatusEnum::Inited) {
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
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }

            // Copy host memory to device memory
            hipError_t hipRes2 = hipMemcpyHtoDAsync(rocm_input_memory, src, input_size, rocmstream);

            if (hipRes2 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
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
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }

            // Copy device memory to host memory
            hipError_t hipRes5 = hipMemcpyDtoHAsync(dst, rocm_output_memory, output_size, rocmstream);

            if (hipRes5 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }

            // Free memory
            hipFreeAsync(rocm_input_memory, rocmstream);
            hipFreeAsync(rocm_output_memory, rocmstream);

            // Stream synchronization
            hipError_t hipRes4 = hipStreamSynchronize(rocmstream);

            if (hipRes4 != 0) {
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }
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
            if (initROCmStatus != InitROCmStatusEnum::Inited) {
                // use cpu
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
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
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
            }

            // Copy host memory to device memory
            hipError_t hipRes2 = hipMemcpyHtoDAsync(rocm_input_memory, src, input_size, rocmstream);

            if (hipRes2 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
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
                hwc2chwROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);

            if (hipRes3 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
            }

            // Copy device memory to host memory
            hipError_t hipRes5 = hipMemcpyDtoHAsync(dst, rocm_output_memory, output_size, rocmstream);

            if (hipRes5 != 0) {
                hipFreeAsync(rocm_input_memory, rocmstream);
                hipFreeAsync(rocm_output_memory, rocmstream);
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
            }

            // Free memory
            hipFreeAsync(rocm_input_memory, rocmstream);
            hipFreeAsync(rocm_output_memory, rocmstream);

            hipError_t hipRes4 = hipStreamSynchronize(rocmstream);

            if (hipRes4 != 0) {
                cpu::chw2hwc<float, uint8_t, true>(h, w, c, src, dst, alpha); return;
            }
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
                return;
            }
            hipError_t hipRes1 = hipStreamSynchronize(rocmstream);
            if (hipRes1 != 0) {
                return;
            }
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
                hwc2chwROCmFun,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                0, rocmstream, args, nullptr);
            if (hipRes0 != 0) {
                return;
            }
            hipError_t hipRes1 = hipStreamSynchronize(rocmstream);
            if (hipRes1 != 0) {
                return;
            }
            return;
        }

    private:
        static std::string compileROCmWithHIPRTC(const std::string& libraryName, const std::string& rocmSource)
        {
            // dynamic load HIPRTC lib
            auto* dlManager = DynamicLibraryManager::instance();
            auto hiprtcLib = dlManager->loadLibrary(libraryName);
            if (!hiprtcLib)
            {
                std::cerr << "Failed to load HIPRTC library: " << libraryName << std::endl;
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
                std::cerr << "Failed to load HIPRTC functions from: " << libraryName << std::endl;
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
                std::cerr << "hiprtcCreateProgram failed: " << hiprtcGetErrorString_fun(res) << std::endl;
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
                std::cerr << "ROCm Compile error: " << log << std::endl;
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
                std::cerr << "Failed to get current directory on Windows." << std::endl;
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
                std::cerr << "Failed to get current directory on Linux." << std::endl;
                return "";
            }

            std::string executablePath(currentDir);
            auto lastSlash = executablePath.find_last_of("/");
            if (lastSlash != std::string::npos)
            {
                executablePath = executablePath.substr(0, lastSlash);
            }
#endif
            // ROCm version list: v5.0 ~ v6.4.
            // ref: https://rocm.docs.amd.com/en/latest/release/versions.html
            const std::vector<std::string> rocmVersions = {
                "0604", "0603", "0602", "0601", "0600", // driver: amdhip64_6.dll
                "0507", "0506", "0505", "0504", "0502", "0501", "0500"  // driver: amdhip64.dll
            };

#ifdef _WIN32
            for (const auto& version : rocmVersions)
            {   //e.g. hiprtc0602.dll
                std::string libraryName = executablePath + "\\hiprtc" + version + ".dll";
                DWORD fileAttr = GetFileAttributesA(libraryName.c_str());
                if (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY))  // file exists
                {
                    return libraryName;
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
                std::string libraryName = executablePath + "/libhiprtc.so." + version;
                if (access(libraryName.c_str(), F_OK) == 0)  // file exists
                {
                    return libraryName;
                }
            }
#endif

            std::cerr << "No suitable HIPRTC library found in the current executable directory: " << executablePath << std::endl;
            return "";
        }

        static bool initROCmDriverAPI()
        {
#ifdef _WIN32
            const std::string driver_dll = "amdhip64_6.dll"; // ROCm v6.x: amdhip64_6.dll, ROCm v5.x: amdhip64.dll
#else
            const std::string driver_dll = "amdhip64_6.so";
#endif
            auto* dlManager = whyb::DynamicLibraryManager::instance();
            auto driverLib = dlManager->loadLibrary(driver_dll);
            if (!driverLib)
            {
#ifdef _WIN32
                const std::string driver_dll = "amdhip64.dll"; // ROCm v6.x: amdhip64_6.dll, ROCm v5.x: amdhip64.dll
#else
                const std::string driver_dll = "amdhip64.so";
#endif
                driverLib = dlManager->loadLibrary(driver_dll); //try again
                if (!driverLib) {
                    std::cerr << "Failed to load AMD ROCm Driver API library: " << driver_dll << std::endl;
                    return false;
                }
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
                std::cerr << "Failed to load one or more AMD ROCm Driver API functions." << std::endl;
                return false;
            }
            return true;
        }

        static bool initROCmFunctions(std::string& compiledPtxStr)
        {
            hipError_t hipRes = hipInit(0);
            if (hipRes != 0) {
                std::cerr << "hipInit failed with error " << hipRes << std::endl;
                return false;
            }
            hipDevice_t device;
            hipRes = hipDeviceGet(&device, 0);
            if (hipRes != 0) {
                std::cerr << "hipDeviceGet failed with error " << hipRes << std::endl;
                return false;
            }
            hipRes = hipCtxCreate(&context, 0, device);
            if (hipRes != 0) {
                std::cerr << "hipCtxCreate failed with error " << hipRes << std::endl;
                return false;
            }
            hipRes = hipStreamCreate(&rocmstream);
            if (hipRes != 0) {
                std::cerr << "hipStreamCreate failed with error " << hipRes << std::endl;
                return false;
            }

            // Load Code(like PTX) module to GPU memory
            hipRes = hipModuleLoadDataEx(&rocmmodule, compiledPtxStr.c_str(), 0, nullptr, nullptr);
            if (hipRes != 0) {
                std::cerr << "hipModuleLoadDataEx failed with error " << hipRes << std::endl;
                hipCtxDestroy(context);
                hipStreamDestroy(rocmstream);
                return false;
            }

            // Get ROCm module kernel function(rocm_hwc2chw)
            hipRes = hipModuleGetFunction(&hwc2chwROCmFun, rocmmodule, "rocm_hwc2chw");
            if (hipRes != 0) {
                std::cerr << "hipModuleGetFunction (rocm_hwc2chw) failed with error " << hipRes << std::endl;
                hipModuleUnload(rocmmodule);
                hipCtxDestroy(context);
                hipStreamDestroy(rocmstream);
                return false;
            }
            // Get ROCm module kernel function(rocm_chw2hwc)
            hipRes = hipModuleGetFunction(&chw2hwcROCmFun, rocmmodule, "rocm_chw2hwc");
            if (hipRes != 0) {
                std::cerr << "hipModuleGetFunction (rocm_chw2hwc) failed with error " << hipRes << std::endl;
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
            if (initROCmStatus == InitROCmStatusEnum::Ready) {
                std::string hiprtc_module_filename = findHIPRTCModuleName();
                if (hiprtc_module_filename.empty()) {
                    std::cerr << "Could not found AMD ROCm HIPRTC dll failed." << std::endl;
                    lastROCmErrorStr = "Could not found AMD ROCm HIPRTC dll failed.";
                    initROCmStatus = InitROCmStatusEnum::Failed;
                    return false;
                }
                std::string code_str = compileROCmWithHIPRTC(hiprtc_module_filename, rocmSource);
                if (code_str.empty()) {
                    std::cerr << "Compile ROCm Source code failed." << std::endl;
                    lastROCmErrorStr = "Compile ROCm Source code failed.";
                    initROCmStatus = InitROCmStatusEnum::Failed;
                    return false;
                }
                bool init_rocm_driver = initROCmDriverAPI();
                if (!init_rocm_driver) {
                    std::cerr << "Failed to load ROCm Driver API functions." << std::endl;
                    lastROCmErrorStr = "Failed to load ROCm Driver API functions.";
                    initROCmStatus = InitROCmStatusEnum::Failed;
                    return false;
                }
                bool init_rocm_functions = initROCmFunctions(code_str);
                if (!init_rocm_functions) {
                    std::cerr << "Failed to load ROCm Driver API functions." << std::endl;
                    lastROCmErrorStr = "Failed to load ROCm Driver API functions.";
                    initROCmStatus = InitROCmStatusEnum::Failed;
                    return false;
                }
                initROCmStatus = InitROCmStatusEnum::Inited;
                return true;
            }
            else if (initROCmStatus == InitROCmStatusEnum::Inited) {
                return true;
            }
            else if (initROCmStatus == InitROCmStatusEnum::Failed) {
                std::cerr << "Init Failed. Last error: " << lastROCmErrorStr << std::endl;
                return false;
            }
            return true;
        }

        static bool releaseAll()
        {
            hipError_t hipRes = hipModuleUnload(rocmmodule);
            if (hipRes != 0) {
                std::cerr << "hipModuleUnload failed with error " << hipRes << std::endl;
                return false;
            }
            hipRes = hipStreamDestroy(rocmstream);
            if (hipRes != 0) {
                std::cerr << "hipStreamDestroy failed with error " << hipRes << std::endl;
                return false;
            }
            hipRes = hipCtxDestroy(context);
            if (hipRes != 0) {
                std::cerr << "hipCtxDestroy failed with error " << hipRes << std::endl;
                return false;
            }
            auto* dlManager = whyb::DynamicLibraryManager::instance();
#ifdef _WIN32
            const std::string driver_dll = "amdhip64_6.dll"; // ROCm v6.x: amdhip64_6.dll, ROCm v5.x: amdhip64.dll
#else
            const std::string driver_dll = "amdhip64_6.so";
#endif
            dlManager->unloadLibrary(driver_dll);
            return true;
        }

    private:
        inline static InitROCmStatusEnum initROCmStatus = InitROCmStatusEnum::Ready;
        inline static std::string lastROCmErrorStr = "";
        inline static std::mutex ROCmMutex;

        inline static hipFunction_t hwc2chwROCmFun = nullptr;
        inline static hipFunction_t chw2hwcROCmFun = nullptr;
        inline static hipCtx_t context;
        inline static hipStream_t rocmstream = nullptr;
        inline static hipModule_t rocmmodule = nullptr;
    };

} // namespace whyb