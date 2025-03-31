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

#include <vector>
#include <string>
#include <filesystem>

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
typedef unsigned long long CUdeviceptr;

// NVIDIA CUDA Driver API function type define
typedef CUresult(*cuInit_t)(unsigned int);
typedef CUresult(*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult(*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(*cuCtxDestroy_t)(CUcontext);
typedef CUresult(*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult(*cuModuleUnload_t)(CUmodule);
typedef CUresult(*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult(*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUcontext, void**, void**);
typedef CUresult(*cuCtxSynchronize_t)(void);
typedef CUresult(*cuMemAlloc_t)(CUdeviceptr*, unsigned int);
typedef CUresult(*cuMemFree_t)(CUdeviceptr);
typedef CUresult(*cuMemcpyHtoD_t)(CUdeviceptr, const void*, unsigned int);
typedef CUresult(*cuMemcpyDtoH_t)(void*, CUdeviceptr, unsigned int);

#ifdef _WIN32
#define DYNAMIC_LIBRARY_EXTENSION ".dll"
#else
#define DYNAMIC_LIBRARY_EXTENSION ".so"
#endif

namespace whyb {

static inline std::string compileCUDAWithNVRTC(const std::string& libraryName, const std::string& cudaSource)
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
    auto nvrtcCreateProgram_fun = reinterpret_cast<nvrtcCreateProgram_t>(dlManager->getFunction(libraryName, "nvrtcCreateProgram"));
    auto nvrtcCompileProgram_fun = reinterpret_cast<nvrtcCompileProgram_t>(dlManager->getFunction(libraryName, "nvrtcCompileProgram"));
    auto nvrtcGetPTXSize_fun = reinterpret_cast<nvrtcGetPTXSize_t>(dlManager->getFunction(libraryName, "nvrtcGetPTXSize"));
    auto nvrtcGetPTX_fun = reinterpret_cast<nvrtcGetPTX_t>(dlManager->getFunction(libraryName, "nvrtcGetPTX"));
    auto nvrtcDestroyProgram_fun = reinterpret_cast<nvrtcDestroyProgram_t>(dlManager->getFunction(libraryName, "nvrtcDestroyProgram"));
    auto nvrtcGetProgramLogSize_fun = reinterpret_cast<nvrtcGetProgramLogSize_t>(dlManager->getFunction(libraryName, "nvrtcGetProgramLogSize"));
    auto nvrtcGetProgramLog_fun = reinterpret_cast<nvrtcGetProgramLog_t>(dlManager->getFunction(libraryName, "nvrtcGetProgramLog"));
    auto nvrtcGetErrorString_fun = reinterpret_cast<nvrtcGetErrorString_t>(dlManager->getFunction(libraryName, "nvrtcGetErrorString"));

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
    nvrtcResult res = nvrtcCreateProgram_fun(&prog, cudaSource.c_str(), "kernel.cu", 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS)
    {
        std::cerr << "nvrtcCreateProgram failed: " << nvrtcGetErrorString_fun(res) << std::endl;
        dlManager->unloadLibrary(libraryName);
        return "";
    }
    std::cout << "Create Program successfully." << std::endl;

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
    std::cout << "Compile CUDA successfully." << std::endl;

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

static inline std::string findNVRTCModuleName()
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
    // CUDA version list: 12.8 ~ 10.0
    const std::vector<std::string> cudaVersions = {
        "128_0", "127_0", "126_0", "125_0", "124_0", "123_0", "122_0", "121_0", "120_0",
        "118_0", "117_0", "116_0", "115_0", "114_0", "113_0", "112_0", "111_0", "110_0",
        "102_0", "101_0", "100_0"
    };

#ifdef _WIN32
    for (const auto& version : cudaVersions)
    {
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
    {
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

} // namespace whyb