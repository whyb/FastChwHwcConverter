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

// AMD ROCm Driver API function type define
typedef hipError_t(*hipInit_t)(unsigned int);
typedef hipError_t(*hipDeviceGet_t)(hipDevice_t*, int);
typedef hipError_t(*hipCtxCreate_t)(hipCtx_t*, unsigned int, hipDevice_t);
typedef hipError_t(*hipCtxDestroy_t)(hipCtx_t);
typedef hipError_t(*hipModuleLoadDataEx_t)(hipModule_t*, const void*, unsigned int, int*, void**);
typedef hipError_t(*hipModuleUnload_t)(hipModule_t);
typedef hipError_t(*hipModuleGetFunction_t)(hipFunction_t*, hipModule_t, const char*);
typedef hipError_t(*hipLaunchKernel_t)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipCtx_t, void**, void**);
typedef hipError_t(*hipCtxSynchronize_t)(void);
//typedef hipError_t(*hipMemAlloc_t)(hipDeviceptr_t*, size_t);
typedef hipError_t(*hipMalloc_t)(hipDeviceptr_t*, size_t);
//typedef hipError_t(*hipMemFree_t)(hipDeviceptr_t);
typedef hipError_t(*hipFree_t)(hipDeviceptr_t);

typedef hipError_t(*hipMemcpyHtoD_t)(hipDeviceptr_t, const void*, size_t);
typedef hipError_t(*hipMemcpyDtoH_t)(void*, hipDeviceptr_t, size_t);

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
static hipModuleLoadDataEx_t hipModuleLoadDataEx = nullptr;
static hipModuleUnload_t hipModuleUnload = nullptr;
static hipModuleGetFunction_t hipModuleGetFunction = nullptr;
static hipLaunchKernel_t hipLaunchKernel = nullptr;
static hipCtxSynchronize_t hipCtxSynchronize = nullptr;
//static hipMemAlloc_t hipMemAlloc = nullptr;
static hipMalloc_t hipMalloc = nullptr;
//static hipMemFree_t hipMemFree = nullptr;
static hipFree_t hipFree = nullptr;
static hipMemcpyHtoD_t hipMemcpyHtoD = nullptr;
static hipMemcpyDtoH_t hipMemcpyDtoH = nullptr;

static const char* rocmSource = R"(
typedef unsigned char uint8_t;

    // HWC -> CHW
    extern "C" __global__ void rocm_hwc2chw(const size_t h, const size_t w, const size_t c,
                                            const uint8_t* src, float* dst, const float alpha = 1.0f) {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;

        if (dx < w && dy < h) {
            for (size_t channel = 0; channel < c; ++channel) {
                size_t src_idx = dy * w * c + dx * c + channel;
                size_t dst_idx = channel * w * h + dy * w + dx;
                dst[dst_idx] = src[src_idx] * alpha;
            }
        }
    }

    // CHW -> HWC
    extern "C" __global__ void rocm_chw2hwc(const size_t c, const size_t h, const size_t w,
                                            const float* src, uint8_t* dst, const uint8_t alpha = 1) {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;

        if (dx < w && dy < h) {
            for (size_t channel = 0; channel < c; ++channel) {
                size_t src_idx = channel * w * h + dy * w + dx;
                size_t dst_idx = dy * w * c + dx * c + channel;
                dst[dst_idx] = static_cast<uint8_t>(src[src_idx] * alpha);
            }
        }
    }

)";

static hipFunction_t hwc2chwFun = nullptr;
static hipFunction_t chw2hwcFun = nullptr;

enum struct InitStatusEnum : int
{
    Ready = 0,
    Inited = 1,
    Failed = 2,
};
static InitStatusEnum initStatus = InitStatusEnum::Ready;
static std::string lastErrorStr = "";
static std::mutex mutex;

namespace whyb {

static inline std::string compileROCmWithHIPRTC(const std::string& libraryName, const std::string& rocmSource)
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
    const char* headers[] = {0};

    const char* includeNames[] = {0};
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

static inline std::string findHIPRTCModuleName()
{
    char currentDir[512] = {0};
#ifdef _WIN32
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
    // ROCm version list: v6.3 ~ 5.0.
    // ref: https://rocm.docs.amd.com/en/latest/release/versions.html
    const std::vector<std::string> rocmVersions = {
        "0603", "0602", "0601", "0600", // driver: amdhip64_6.dll
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

static inline bool initROCmDriverAPI()
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
        std::cerr << "Failed to load AMD ROCm Driver API library: " << driver_dll << std::endl;
        return false;
    }
    hipInit = (hipInit_t)(dlManager->getFunction(driver_dll, "hipInit"));
    hipDeviceGet = (hipDeviceGet_t)(dlManager->getFunction(driver_dll, "hipDeviceGet"));
    hipCtxCreate = (hipCtxCreate_t)(dlManager->getFunction(driver_dll, "hipCtxCreate"));
    hipCtxDestroy = (hipCtxDestroy_t)(dlManager->getFunction(driver_dll, "hipCtxDestroy"));
    hipModuleLoadDataEx = (hipModuleLoadDataEx_t)(dlManager->getFunction(driver_dll, "hipModuleLoadDataEx"));
    hipModuleUnload = (hipModuleUnload_t)(dlManager->getFunction(driver_dll, "hipModuleUnload"));
    hipModuleGetFunction = (hipModuleGetFunction_t)(dlManager->getFunction(driver_dll, "hipModuleGetFunction"));
    hipLaunchKernel = (hipLaunchKernel_t)(dlManager->getFunction(driver_dll, "hipLaunchKernel"));
    hipCtxSynchronize = (hipCtxSynchronize_t)(dlManager->getFunction(driver_dll, "hipCtxSynchronize"));
    //hipMemAlloc = (hipMemAlloc_t)(dlManager->getFunction(driver_dll, "hipMemAlloc"));
    hipMalloc = (hipMalloc_t)(dlManager->getFunction(driver_dll, "hipMalloc")); // == hipMemAlloc
    //hipMemFree = (hipMemFree_t)(dlManager->getFunction(driver_dll, "hipMemFree"));
    hipFree = (hipFree_t)(dlManager->getFunction(driver_dll, "hipFree")); // == hipMemFree
    hipMemcpyHtoD = (hipMemcpyHtoD_t)(dlManager->getFunction(driver_dll, "hipMemcpyHtoD"));
    hipMemcpyDtoH = (hipMemcpyDtoH_t)(dlManager->getFunction(driver_dll, "hipMemcpyDtoH"));

    if (!hipInit || !hipDeviceGet || !hipCtxCreate || !hipModuleLoadDataEx ||
        !hipModuleGetFunction || !hipLaunchKernel || !hipCtxSynchronize ||
        !hipMalloc || !hipFree || !hipMemcpyHtoD || !hipMemcpyDtoH) {
        std::cerr << "Failed to load one or more AMD ROCm Driver API functions." << std::endl;
        return false;
    }
    return true;
}

static inline bool initROCmFunctions(std::string& compiledPtxStr)
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
    hipCtx_t context;
    hipRes = hipCtxCreate(&context, 0, device);
    if (hipRes != 0) {
        std::cerr << "hipCtxCreate failed with error " << hipRes << std::endl;
        return false;
    }

    // 加载编译好的 PTX 模块到 GPU 内存中
    hipModule_t module;
    hipRes = hipModuleLoadDataEx(&module, compiledPtxStr.c_str(), 0, nullptr, nullptr);
    if (hipRes != 0) {
        std::cerr << "hipModuleLoadDataEx failed with error " << hipRes << std::endl;
        hipCtxDestroy(context);
        return false;
    }

    // Get ROCm module kernel function(rocm_hwc2chw)
    hipRes = hipModuleGetFunction(&hwc2chwFun, module, "rocm_hwc2chw");
    if (hipRes != 0) {
        std::cerr << "hipModuleGetFunction (rocm_hwc2chw) failed with error " << hipRes << std::endl;
        hipModuleUnload(module);
        hipCtxDestroy(context);
        return false;
    }
    // Get ROCm module kernel function(rocm_chw2hwc)
    hipRes = hipModuleGetFunction(&chw2hwcFun, module, "rocm_chw2hwc");
    if (hipRes != 0) {
        std::cerr << "hipModuleGetFunction (rocm_chw2hwc) failed with error " << hipRes << std::endl;
        hipModuleUnload(module);
        hipCtxDestroy(context);
        return false;
    }
    return true;
}

static inline bool initAll()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (initStatus == InitStatusEnum::Ready) {
        std::string hiprtc_module_filename = findHIPRTCModuleName();
        if (hiprtc_module_filename.empty()) {
            std::cerr << "Could not found AMD ROCm HIPRTC dll failed." << std::endl;
            lastErrorStr = "Could not found AMD ROCm HIPRTC dll failed.";
            initStatus = InitStatusEnum::Failed;
            return false;
        }
        std::string code_str = compileROCmWithHIPRTC(hiprtc_module_filename, rocmSource);
        if (code_str.empty()) {
            std::cerr << "Compile ROCm Source code failed." << std::endl;
            lastErrorStr = "Compile ROCm Source code failed.";
            initStatus = InitStatusEnum::Failed;
            return false;
        }
        bool init_rocm_driver = initROCmDriverAPI();
        if (!init_rocm_driver) {
            std::cerr << "Failed to load ROCm Driver API functions." << std::endl;
            lastErrorStr = "Failed to load ROCm Driver API functions.";
            initStatus = InitStatusEnum::Failed;
            return false;
        }
        bool init_rocm_functions = initROCmFunctions(code_str);
        if (!init_rocm_functions) {
            std::cerr << "Failed to load ROCm Driver API functions." << std::endl;
            lastErrorStr = "Failed to load ROCm Driver API functions.";
            initStatus = InitStatusEnum::Failed;
            return false;
        }
        initStatus = InitStatusEnum::Inited;
        return true;
    }
    else if (initStatus == InitStatusEnum::Inited) {
        return true;
    }
    else if (initStatus == InitStatusEnum::Failed) {
        std::cerr << "Init Failed. Last error: " << lastErrorStr << std::endl;
        return false;
    }
}

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
inline void hwc2chw_rocm(
    const size_t h, const size_t w, const size_t c,
    const uint8_t* src, float* dst,
    const float alpha = 1.f/255.f) {
    if (!initAll()) {
        // use cpu
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    // use rocm
    const size_t pixel_size = h * w * c;
    const size_t input_size = pixel_size * sizeof(uint8_t);
    const size_t output_size = pixel_size * sizeof(float);
    hipDeviceptr_t rocm_input_memory = 0;
    hipDeviceptr_t rocm_output_memory = 0;
    // alloc device memory
    hipError_t hipRes0 = hipMalloc(&rocm_input_memory, input_size);
    hipError_t hipRes1 = hipMalloc(&rocm_output_memory, output_size);
    if (hipRes0 != 0 || hipRes1 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    // copy host memory to device memory
    hipError_t hipRes2 = hipMemcpyHtoD(rocm_input_memory, src, input_size);
    if (hipRes2 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    // call rocm function
    if (hwc2chwFun == nullptr) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    const unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
    const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
    const unsigned int gridDimZ = 1;
    // for ready rocm kernel function(func_hwc2chw)
    size_t arg_h_val = h;
    size_t arg_w_val = w;
    size_t arg_c_val = c;
    float arg_alpha_val = alpha;
    void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &rocm_input_memory, &rocm_output_memory, &arg_alpha_val };
    hipError_t hipRes3 = hipLaunchKernel(
        hwc2chwFun, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args1, nullptr);
    if (hipRes3 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    hipError_t hipRes4 = hipCtxSynchronize();
    if (hipRes4 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    // copy device memory to host memory
    hipError_t hipRes5 = hipMemcpyDtoH(dst, rocm_output_memory, output_size);
    if (hipRes5 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        hwc2chw<uint8_t, float>(h, w, c, src, dst, alpha); return;
    }
    hipFree(rocm_input_memory);
    hipFree(rocm_output_memory);
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
inline void chw2hwc_rocm(
    const size_t c, const size_t h, const size_t w,
    const float* src, uint8_t* dst,
    const uint8_t alpha = 255.0f) {
    if (!initAll()) {
        // use cpu
        chw2hwc<float, uint8_t>(c, h, w, src, dst, alpha); return;
    }
    // use rocm
    const size_t pixel_size = h * w * c;
    const size_t input_size = pixel_size * sizeof(float);
    const size_t output_size = pixel_size * sizeof(uint8_t);
    hipDeviceptr_t rocm_input_memory = 0;
    hipDeviceptr_t rocm_output_memory = 0;
    // alloc device memory
    hipError_t hipRes0 = hipMalloc(&rocm_input_memory, input_size);
    hipError_t hipRes1 = hipMalloc(&rocm_output_memory, output_size);
    if (hipRes0 != 0 || hipRes1 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    // copy host memory to device memory
    hipError_t hipRes2 = hipMemcpyHtoD(rocm_input_memory, src, input_size);
    if (hipRes2 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    // call rocm function
    if (chw2hwcFun == nullptr) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    const unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
    const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
    const unsigned int gridDimZ = 1;
    // for ready rocm kernel function(func_hwc2chw)
    size_t arg_c_val = c;
    size_t arg_h_val = h;
    size_t arg_w_val = w;
    uint8_t arg_alpha_val = alpha;
    void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &rocm_input_memory, &rocm_output_memory, &arg_alpha_val };
    hipError_t hipRes3 = hipLaunchKernel(
        chw2hwcFun, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args, nullptr);
    if (hipRes3 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    hipError_t hipRes4 = hipCtxSynchronize();
    if (hipRes4 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    // copy device memory to host memory
    hipError_t hipRes5 = hipMemcpyDtoH(dst, rocm_output_memory, output_size);
    if (hipRes5 != 0) {
        hipFree(rocm_input_memory);
        hipFree(rocm_output_memory);
        chw2hwc<float, uint8_t>(h, w, c, src, dst, alpha); return;
    }
    hipError_t hipRes6 = hipFree(rocm_input_memory);
    hipError_t hipRes7 = hipFree(rocm_output_memory);
    return;
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
inline void hwc2chw_rocm(
    const size_t h, const size_t w, const size_t c,
    hipDeviceptr_t src, hipDeviceptr_t dst,
    const float alpha = 1.f / 255.f) {

    const size_t pixel_size = h * w * c;
    const size_t input_size = pixel_size * sizeof(uint8_t);
    const size_t output_size = pixel_size * sizeof(float);

    const unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
    const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
    const unsigned int gridDimZ = 1;
    // for ready rocm kernel function(func_hwc2chw)
    size_t arg_h_val = h;
    size_t arg_w_val = w;
    size_t arg_c_val = c;
    float arg_alpha_val = alpha;
    void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &src, &dst, &arg_alpha_val };
    hipError_t hipRes0 = hipLaunchKernel(
        hwc2chwFun, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args1, nullptr);
    if (hipRes0 != 0) {
        return;
    }
    hipError_t hipRes1 = hipCtxSynchronize();
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
inline void chw2hwc_rocm(
    const size_t c, const size_t h, const size_t w,
    hipDeviceptr_t src, hipDeviceptr_t dst,
    const uint8_t alpha = 255.0f) {
    
    const unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    const unsigned int gridDimX = ((unsigned int)w + blockDimX - 1) / blockDimX;
    const unsigned int gridDimY = ((unsigned int)h + blockDimY - 1) / blockDimY;
    const unsigned int gridDimZ = 1;
    // for ready rocm kernel function(func_hwc2chw)
    size_t arg_c_val = c;
    size_t arg_h_val = h;
    size_t arg_w_val = w;
    uint8_t arg_alpha_val = alpha;
    void* args[] = { &arg_c_val, &arg_h_val, &arg_w_val, &src, &dst, &arg_alpha_val };
    hipError_t hipRes0 = hipLaunchKernel(
        chw2hwcFun, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args, nullptr);
    if (hipRes0 != 0) {
        return;
    }
    hipError_t hipRes1 = hipCtxSynchronize();
    if (hipRes1 != 0) {
        return;
    }
    return;
}

} // namespace whyb