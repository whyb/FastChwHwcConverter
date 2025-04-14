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
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif


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
typedef CUresult(*cuMemAllocAsync_t)(CUdeviceptr*, size_t, CUstream);
typedef CUresult(*cuMemAllocHost_t)(void**, size_t);
typedef CUresult(*cuMemFree_t)(CUdeviceptr);
typedef CUresult(*cuMemFreeAsync_t)(CUdeviceptr, CUstream);
typedef CUresult(*cuMemFreeHost_t)(void*);

typedef CUresult(*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult(*cuMemcpyHtoDAsync_t)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult(*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult(*cuMemcpyDtoHAsync_t)(void*, CUdeviceptr, size_t, CUstream);


#ifdef _WIN32
#define DYNAMIC_LIBRARY_EXTENSION ".dll"
#else
#define DYNAMIC_LIBRARY_EXTENSION ".so"
#endif

// static Cuda Driver API function points
static cuInit_t cuInit = nullptr;
static cuDeviceGet_t cuDeviceGet = nullptr;
static cuCtxCreate_t cuCtxCreate = nullptr;
static cuCtxDestroy_t cuCtxDestroy = nullptr;
static cuStreamCreate_t cuStreamCreate = nullptr;
static cuStreamDestroy_t cuStreamDestroy = nullptr;
static cuStreamSynchronize_t cuStreamSynchronize = nullptr;
static cuModuleLoadDataEx_t cuModuleLoadDataEx = nullptr;
static cuModuleUnload_t cuModuleUnload = nullptr;
static cuModuleGetFunction_t cuModuleGetFunction = nullptr;
static cuLaunchKernel_t cuLaunchKernel = nullptr;
static cuCtxSynchronize_t cuCtxSynchronize = nullptr;
static cuMemAlloc_t cuMemAlloc = nullptr;
static cuMemAllocAsync_t cuMemAllocAsync = nullptr;
static cuMemAllocHost_t cuMemAllocHost = nullptr;
static cuMemFree_t cuMemFree = nullptr;
static cuMemFreeAsync_t cuMemFreeAsync = nullptr;
static cuMemFreeHost_t cuMemFreeHost = nullptr;
static cuMemcpyHtoD_t cuMemcpyHtoD = nullptr;
static cuMemcpyHtoDAsync_t cuMemcpyHtoDAsync = nullptr;
static cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;
static cuMemcpyDtoHAsync_t cuMemcpyDtoHAsync = nullptr;

static const char* cudaSource = R"(
  typedef unsigned char uint8_t;

  // HWC -> CHW
  extern "C" __global__ void cuda_hwc2chw(const size_t h, const size_t w, const size_t c,
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
   extern "C" __global__ void cuda_chw2hwc(const size_t c, const size_t h, const size_t w,
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

enum struct InitCUDAStatusEnum : int
{
    Ready = 0,
    Inited = 1,
    Failed = 2,
};

namespace whyb {
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
            if (initCUDAStatus != InitCUDAStatusEnum::Inited) {
                // use cpu
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
            // use cuda
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(uint8_t);
            const size_t output_size = pixel_size * sizeof(float);
            CUdeviceptr cuda_input_memory = 0;
            CUdeviceptr cuda_output_memory = 0;
            // alloc device memory
            CUresult cuRes0 = cuMemAllocAsync(&cuda_input_memory, input_size, cudastream);
            CUresult cuRes1 = cuMemAllocAsync(&cuda_output_memory, output_size, cudastream);
            if (cuRes0 != 0 || cuRes1 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
            // copy host memory to device memory
            CUresult cuRes2 = cuMemcpyHtoD(cuda_input_memory, src, input_size);
            if (cuRes2 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
            // call cuda function
            if (hwc2chwCUDAFun == nullptr) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
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
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
            // copy device memory to host memory
            CUresult cuRes4 = cuMemcpyDtoHAsync(dst, cuda_output_memory, output_size, cudastream);
            if (cuRes4 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
            CUresult cuRes5 = cuMemFreeAsync(cuda_input_memory, cudastream);
            CUresult cuRes6 = cuMemFreeAsync(cuda_output_memory, cudastream);
            CUresult cuRes7 = cuStreamSynchronize(cudastream);
            if (cuRes7 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
            }
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
            if (initCUDAStatus != InitCUDAStatusEnum::Inited) {
                // use cpu
                cpu::chw2hwc<float, uint8_t>(c, h, w, src, dst, alpha); return;
            }
            // use cuda
            const size_t pixel_size = h * w * c;
            const size_t input_size = pixel_size * sizeof(float);
            const size_t output_size = pixel_size * sizeof(uint8_t);
            CUdeviceptr cuda_input_memory = 0;
            CUdeviceptr cuda_output_memory = 0;
            // alloc device memory
            CUresult cuRes0 = cuMemAllocAsync(&cuda_input_memory, input_size, cudastream);
            CUresult cuRes1 = cuMemAllocAsync(&cuda_output_memory, output_size, cudastream);
            if (cuRes0 != 0 || cuRes1 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
            }
            // copy host memory to device memory
            CUresult cuRes2 = cuMemcpyHtoDAsync(cuda_input_memory, src, input_size, cudastream);
            if (cuRes2 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
            }
            // call cuda function
            if (chw2hwcCUDAFun == nullptr) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
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
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
            }
            // copy device memory to host memory
            CUresult cuRes4 = cuMemcpyDtoHAsync(dst, cuda_output_memory, output_size, cudastream);
            if (cuRes4 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
            }
            CUresult cuRes5 = cuMemFreeAsync(cuda_input_memory, cudastream);
            CUresult cuRes6 = cuMemFreeAsync(cuda_output_memory, cudastream);
            CUresult cuRes7 = cuStreamSynchronize(cudastream);
            if (cuRes7 != 0) {
                cuMemFreeAsync(cuda_input_memory, cudastream);
                cuMemFreeAsync(cuda_output_memory, cudastream);
                cpu::chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
            }
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
                return;
            }
            CUresult cuRes1 = cuCtxSynchronize();
            if (cuRes1 != 0) {
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
        * @param src Cuda Memory (float) Pointer to the source data in CHW format
        * @param dst Cuda Memory (uint8_t) Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            CUdeviceptr src, CUdeviceptr dst,
            const uint8_t alpha = 255.0f) {
            nvidia();
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
                return;
            }
            CUresult cuRes1 = cuCtxSynchronize();
            if (cuRes1 != 0) {
                return;
            }
            return;
        }
    private:
        static std::string compileCUDAWithNVRTC(const std::string& libraryName, const std::string& cudaSource)
        {
            // dynamic load NVRTC lib
            auto* dlManager = DynamicLibraryManager::instance();
            auto nvrtcLib = dlManager->loadLibrary(libraryName);
            if (!nvrtcLib)
            {
                std::cerr << "Failed to load NVRTC library: " << libraryName << std::endl;
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
                std::cerr << "Failed to load NVRTC functions from: " << libraryName << std::endl;
                dlManager->unloadLibrary(libraryName);
                return "";
            }

            // Create NVRTC Program Object
            nvrtcProgram prog;
            nvrtcResult res = nvrtcCreateProgram_fun(&prog, cudaSource.c_str(), "FastChwHwcConverterCuda.cu", 0, nullptr, nullptr);
            if (res != NVRTC_SUCCESS)
            {
                std::cerr << "nvrtcCreateProgram failed: " << nvrtcGetErrorString_fun(res) << std::endl;
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
                std::cerr << "CUDA Compile error: " << log << std::endl;
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
            // CUDA version list: 12.8 ~ 10.0
            const std::vector<std::string> cudaVersions = {
                "128_0", "127_0", "126_0", "125_0", "124_0", "123_0", "122_0", "121_0", "120_0",
                "118_0", "117_0", "116_0", "115_0", "114_0", "113_0", "112_0", "111_0", "110_0",
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

            std::cerr << "No suitable NVRTC library found in the current executable directory: " << executablePath << std::endl;
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
                std::cerr << "Failed to load NVIDIA Driver API library: " << driver_dll << std::endl;
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
            cuMemAllocAsync = (cuMemAllocAsync_t)(dlManager->getFunction(driver_dll, "cuMemAllocAsync"));
            cuMemFree = (cuMemFree_t)(dlManager->getFunction(driver_dll, "cuMemFree_v2"));
            cuMemFreeAsync = (cuMemFreeAsync_t)(dlManager->getFunction(driver_dll, "cuMemFreeAsync"));
            cuMemFreeHost = (cuMemFreeHost_t)(dlManager->getFunction(driver_dll, "cuMemFreeHost"));
            cuMemcpyHtoD = (cuMemcpyHtoD_t)(dlManager->getFunction(driver_dll, "cuMemcpyHtoD_v2"));
            cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_t)(dlManager->getFunction(driver_dll, "cuMemcpyHtoDAsync_v2"));
            cuMemcpyDtoH = (cuMemcpyDtoH_t)(dlManager->getFunction(driver_dll, "cuMemcpyDtoH_v2"));
            cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_t)(dlManager->getFunction(driver_dll, "cuMemcpyDtoHAsync_v2"));

            if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuModuleLoadDataEx ||
                !cuModuleGetFunction || !cuLaunchKernel || !cuCtxSynchronize ||
                !cuStreamCreate || !cuStreamDestroy || !cuStreamSynchronize ||
                !cuMemAlloc || !cuMemAllocAsync || !cuMemAllocHost ||
                !cuMemFree || !cuMemFreeAsync || !cuMemFreeHost ||
                !cuMemcpyHtoD || !cuMemcpyDtoH || !cuMemcpyHtoDAsync || !cuMemcpyDtoHAsync) {
                std::cerr << "Failed to load one or more CUDA Driver API functions." << std::endl;
                return false;
            }
            return true;
        }

        static bool initCudaFunctions(std::string& compiledPtxStr)
        {
            CUresult cuRes = cuInit(0);
            if (cuRes != 0) {
                std::cerr << "cuInit failed with error " << cuRes << std::endl;
                return false;
            }
            CUdevice device;
            cuRes = cuDeviceGet(&device, 0);
            if (cuRes != 0) {
                std::cerr << "cuDeviceGet failed with error " << cuRes << std::endl;
                return false;
            }
            cuRes = cuCtxCreate(&context, 0, device);
            if (cuRes != 0) {
                std::cerr << "cuCtxCreate failed with error " << cuRes << std::endl;
                return false;
            }
            cuRes = cuStreamCreate(&cudastream, 0); //flag: CU_STREAM_DEFAULT = 0
            if (cuRes != 0) {
                std::cerr << "cuStreamCreate failed with error " << cuRes << std::endl;
                return false;
            }

            // Load PTX module to GPU Memory
            cuRes = cuModuleLoadDataEx(&cudamodule, compiledPtxStr.c_str(), 0, nullptr, nullptr);
            if (cuRes != 0) {
                std::cerr << "cuModuleLoadDataEx failed with error " << cuRes << std::endl;
                cuCtxDestroy(context);
                cuStreamDestroy(cudastream);
                return false;
            }

            // Get cuda module kernel function(cuda_hwc2chw)
            cuRes = cuModuleGetFunction(&hwc2chwCUDAFun, cudamodule, "cuda_hwc2chw");
            if (cuRes != 0) {
                std::cerr << "cuModuleGetFunction (cuda_hwc2chw) failed with error " << cuRes << std::endl;
                cuModuleUnload(cudamodule);
                cuCtxDestroy(context);
                cuStreamDestroy(cudastream);
                return false;
            }
            // Get cuda module kernel function(cuda_chw2hwc)
            cuRes = cuModuleGetFunction(&chw2hwcCUDAFun, cudamodule, "cuda_chw2hwc");
            if (cuRes != 0) {
                std::cerr << "cuModuleGetFunction (cuda_chw2hwc) failed with error " << cuRes << std::endl;
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
            if (initCUDAStatus == InitCUDAStatusEnum::Ready) {
                std::string nvrtc_module_filename = findNVRTCModuleName();
                if (nvrtc_module_filename.empty()) {
                    std::cerr << "Could not found CUDA NVRTC dll failed." << std::endl;
                    lastCUDAErrorStr = "Could not found CUDA NVRTC dll failed.";
                    initCUDAStatus = InitCUDAStatusEnum::Failed;
                    return false;
                }
                std::string ptx_str = compileCUDAWithNVRTC(nvrtc_module_filename, cudaSource);
                if (ptx_str.empty()) {
                    std::cerr << "Compile CUDA Source code failed." << std::endl;
                    lastCUDAErrorStr = "Compile CUDA Source code failed.";
                    initCUDAStatus = InitCUDAStatusEnum::Failed;
                    return false;
                }
                bool init_cuda_driver = initCudaDriverAPI();
                if (!init_cuda_driver) {
                    std::cerr << "Failed to load CUDA Driver API functions." << std::endl;
                    lastCUDAErrorStr = "Failed to load CUDA Driver API functions.";
                    initCUDAStatus = InitCUDAStatusEnum::Failed;
                    return false;
                }
                bool init_cuda_functions = initCudaFunctions(ptx_str);
                if (!init_cuda_functions) {
                    std::cerr << "Failed to load CUDA Driver API functions." << std::endl;
                    lastCUDAErrorStr = "Failed to load CUDA Driver API functions.";
                    initCUDAStatus = InitCUDAStatusEnum::Failed;
                    return false;
                }
                initCUDAStatus = InitCUDAStatusEnum::Inited;
                return true;
            }
            else if (initCUDAStatus == InitCUDAStatusEnum::Inited) {
                return true;
            }
            else if (initCUDAStatus == InitCUDAStatusEnum::Failed) {
                std::cerr << "Init Failed. Last error: " << lastCUDAErrorStr << std::endl;
                return false;
            }
            return true;
        }

        static bool releaseAll()
        {
            CUresult cuRes = cuModuleUnload(cudamodule);
            if (cuRes != 0) {
                std::cerr << "hipModuleUnload failed with error " << cuRes << std::endl;
                return false;
            }
            cuRes = cuStreamDestroy(cudastream);
            if (cuRes != 0) {
                std::cerr << "hipStreamDestroy failed with error " << cuRes << std::endl;
                return false;
            }
            cuRes = cuCtxDestroy(context);
            if (cuRes != 0) {
                std::cerr << "hipCtxDestroy failed with error " << cuRes << std::endl;
                return false;
            }
            return true;
        }
    private:
        inline static InitCUDAStatusEnum initCUDAStatus = InitCUDAStatusEnum::Ready;
        inline static std::string lastCUDAErrorStr = "";
        inline static std::mutex CUDAMutex;

        inline static CUfunction hwc2chwCUDAFun = nullptr;
        inline static CUfunction chw2hwcCUDAFun = nullptr;
        inline static CUcontext context;
        inline static CUstream cudastream = nullptr;
        inline static CUmodule cudamodule = nullptr;
    };

} // namespace whyb