#include <iostream>
#include <windows.h>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include "FastChwHwcConverterCuda.hpp"
#define TEST_COUNT 1000

const char* cudaSource = R"(
typedef unsigned char uint8_t;

    // HWC -> CHW
    extern "C" __global__ void cuda_hwc2chw(const size_t h, const size_t w, const size_t c,
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
    extern "C" __global__ void cuda_chw2hwc(const size_t c, const size_t h, const size_t w,
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

int main() {
    std::string nvrtc_module_filename = whyb::findNVRTCModuleName();
    if (nvrtc_module_filename.empty()) {
        std::cerr << "Could not found CUDA NVRTC dll failed." << std::endl;
        return -1;
    }
    std::string ptx_str = whyb::compileCUDAWithNVRTC(nvrtc_module_filename, cudaSource);
    if (ptx_str.empty()) {
        std::cerr << "Compile CUDA Source code failed." << std::endl;
        return -1;
    }
    std::cout << "Compile CUDA Source code to PTX Successfully." << std::endl;


    //================ 使用 CUDA Driver API 加载 PTX 模块并调用内核 =================
    const char* driver_dll = "nvcuda.dll";
    HMODULE driverLib = LoadLibrary(driver_dll);
    if (!driverLib) {
        std::cerr << "Load CUDA driver dll failed: " << driver_dll << std::endl;
        return -1;
    }
    auto cuInit = (cuInit_t)GetProcAddress(driverLib, "cuInit");
    auto cuDeviceGet = (cuDeviceGet_t)GetProcAddress(driverLib, "cuDeviceGet");
    auto cuCtxCreate = (cuCtxCreate_t)GetProcAddress(driverLib, "cuCtxCreate_v2");
    auto cuCtxDestroy = (cuCtxDestroy_t)GetProcAddress(driverLib, "cuCtxDestroy_v2");
    auto cuModuleLoadDataEx = (cuModuleLoadDataEx_t)GetProcAddress(driverLib, "cuModuleLoadDataEx");
    auto cuModuleUnload = (cuModuleUnload_t)GetProcAddress(driverLib, "cuModuleUnload");
    auto cuModuleGetFunction = (cuModuleGetFunction_t)GetProcAddress(driverLib, "cuModuleGetFunction");
    auto cuLaunchKernel = (cuLaunchKernel_t)GetProcAddress(driverLib, "cuLaunchKernel");
    auto cuCtxSynchronize = (cuCtxSynchronize_t)GetProcAddress(driverLib, "cuCtxSynchronize");
    auto cuMemAlloc = (cuMemAlloc_t)GetProcAddress(driverLib, "cuMemAlloc_v2");
    auto cuMemFree = (cuMemFree_t)GetProcAddress(driverLib, "cuMemFree_v2");
    auto cuMemcpyHtoD = (cuMemcpyHtoD_t)GetProcAddress(driverLib, "cuMemcpyHtoD_v2");
    auto cuMemcpyDtoH = (cuMemcpyDtoH_t)GetProcAddress(driverLib, "cuMemcpyDtoH_v2");

    if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuModuleLoadDataEx ||
        !cuModuleGetFunction || !cuLaunchKernel || !cuCtxSynchronize ||
        !cuMemAlloc || !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH) {
        std::cerr << "Failed to load one or more CUDA Driver API functions." << std::endl;
        FreeLibrary(driverLib);
        return -1;
    }

    CUresult cuRes = cuInit(0);
    if (cuRes != 0) {
        std::cerr << "cuInit failed with error " << cuRes << std::endl;
        FreeLibrary(driverLib);
        return -1;
    }
    CUdevice device;
    cuRes = cuDeviceGet(&device, 0);
    if (cuRes != 0) {
        std::cerr << "cuDeviceGet failed with error " << cuRes << std::endl;
        FreeLibrary(driverLib);
        return -1;
    }
    CUcontext context;
    cuRes = cuCtxCreate(&context, 0, device);
    if (cuRes != 0) {
        std::cerr << "cuCtxCreate failed with error " << cuRes << std::endl;
        FreeLibrary(driverLib);
        return -1;
    }

    // 加载编译好的 PTX 模块到 GPU 内存中
    CUmodule module;
    cuRes = cuModuleLoadDataEx(&module, ptx_str.c_str(), 0, nullptr, nullptr);
    if (cuRes != 0) {
        std::cerr << "cuModuleLoadDataEx failed with error " << cuRes << std::endl;
        cuCtxDestroy(context);
        FreeLibrary(driverLib);
        return -1;
    }

    // 获取两个内核函数句柄
    CUfunction func_hwc2chw;
    cuRes = cuModuleGetFunction(&func_hwc2chw, module, "cuda_hwc2chw");
    if (cuRes != 0) {
        std::cerr << "cuModuleGetFunction (cuda_hwc2chw) failed with error " << cuRes << std::endl;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        FreeLibrary(driverLib);
        return -1;
    }
    CUfunction func_chw2hwc;
    cuRes = cuModuleGetFunction(&func_chw2hwc, module, "cuda_chw2hwc");
    if (cuRes != 0) {
        std::cerr << "cuModuleGetFunction (cuda_chw2hwc) failed with error " << cuRes << std::endl;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        FreeLibrary(driverLib);
        return -1;
    }

    const std::vector<size_t> channels = { 1, 3, 4 };
    const std::vector<std::pair<size_t, size_t>> resolutions = {
        {426, 240},   // 240p  (SD)
        {640, 360},   // 360p  (SD)
        {854, 480},   // 480p  (SD)
        {1280, 720},  // 720p  (HD)
        {1920, 1080}, // 1080p (HD)
        {2560, 1440}, // 1440p (2K)
        {3840, 2160}, // 2160p (4K)
        {7680, 4320}  // 4320p (8K)
    };

    std::cout << "Width,\tHeight,\tChannel,\thwc2chw,\tchw2hwc" << std::endl;

    for (auto& resolution : resolutions) {
        const size_t& width = resolution.first;
        const size_t& height = resolution.second;

        for (auto& channel : channels) {
            // Defining input and output 
            const size_t pixel_size = height * width * channel;
            //std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
            //std::vector<float> src_float(pixel_size); // Source data(chw)
            //
            //std::vector<float> out_float(pixel_size); // Inference output data(chw)
            //std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

            const size_t inputSizeBytes = pixel_size * sizeof(uint8_t);
            const size_t tempSizeBytes = pixel_size * sizeof(float);
            const size_t outputSizeBytes = pixel_size * sizeof(uint8_t);
            CUdeviceptr d_input = 0, d_temp = 0, d_output = 0;

            cuRes = cuMemAlloc(&d_input, inputSizeBytes);
            cuRes = cuMemAlloc(&d_temp, tempSizeBytes);
            cuRes = cuMemAlloc(&d_output, outputSizeBytes);

            unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
            unsigned int gridDimX = (width + blockDimX - 1) / blockDimX;
            unsigned int gridDimY = (height + blockDimY - 1) / blockDimY;
            unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_h_val = static_cast<size_t>(height);
            size_t arg_w_val = static_cast<size_t>(width);
            size_t arg_c_val = static_cast<size_t>(channel);
            float arg_alpha_val = 1.f / 255.f;
            void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &d_input, &d_temp, &arg_alpha_val };

            auto startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                //whyb::hwc2chw<uint8_t, float>(height, width, channel, (uint8_t*)src_uint8.data(), (float*)src_float.data());
                cuRes = cuLaunchKernel(func_hwc2chw, gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ,
                    0, nullptr, args1, nullptr);
            }
            cuRes = cuCtxSynchronize();
            auto endTime = std::chrono::high_resolution_clock::now();
            auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            // for ready cuda kernel function(func_chw2hwc)
            size_t arg_c2_val = static_cast<size_t>(channel);
            size_t arg_h2_val = static_cast<size_t>(height);
            size_t arg_w2_val = static_cast<size_t>(width);
            uint8_t arg_alpha2_val = 255;
            void* args2[] = { &arg_c2_val, &arg_h2_val, &arg_w2_val, &d_temp, &d_output, &arg_alpha2_val };
            startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                //whyb::chw2hwc<float, uint8_t>(channel, height, width, (float*)out_float.data(), (uint8_t*)out_uint8.data());
                cuRes = cuLaunchKernel(func_chw2hwc, gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ,
                    0, nullptr, args2, nullptr);
            }
            cuRes = cuCtxSynchronize();
            endTime = std::chrono::high_resolution_clock::now();
            auto chw2hwcDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << std::fixed << std::setprecision(3)
                << hwc2chwDuration.count() / 1000.0 << "ms,\t"
                << chw2hwcDuration.count() / 1000.0 << "ms" << std::endl;

            cuMemFree(d_input);
            cuMemFree(d_temp);
            cuMemFree(d_output);
        }
    }

cleanup:
    cuModuleUnload(module);
    cuCtxDestroy(context);
    FreeLibrary(driverLib);

    std::cout << "CUDA Benchmark completed successfully!" << std::endl;
    return 0;
}
