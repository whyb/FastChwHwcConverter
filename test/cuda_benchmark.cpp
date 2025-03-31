#include <iostream>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include "FastChwHwcConverterCuda.hpp"
#define TEST_COUNT 1000

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
    bool init_cuda_driver = whyb::initCudaDriverAPI();
    if (!init_cuda_driver) {
        std::cerr << "Failed to load CUDA Driver API functions." << std::endl;
        return -1;
    }

    CUresult cuRes = cuInit(0);
    if (cuRes != 0) {
        std::cerr << "cuInit failed with error " << cuRes << std::endl;
        return -1;
    }
    CUdevice device;
    cuRes = cuDeviceGet(&device, 0);
    if (cuRes != 0) {
        std::cerr << "cuDeviceGet failed with error " << cuRes << std::endl;
        return -1;
    }
    CUcontext context;
    cuRes = cuCtxCreate(&context, 0, device);
    if (cuRes != 0) {
        std::cerr << "cuCtxCreate failed with error " << cuRes << std::endl;
        return -1;
    }

    // 加载编译好的 PTX 模块到 GPU 内存中
    CUmodule module;
    cuRes = cuModuleLoadDataEx(&module, ptx_str.c_str(), 0, nullptr, nullptr);
    if (cuRes != 0) {
        std::cerr << "cuModuleLoadDataEx failed with error " << cuRes << std::endl;
        cuCtxDestroy(context);
        return -1;
    }

    // 获取两个内核函数句柄
    CUfunction func_hwc2chw;
    cuRes = cuModuleGetFunction(&func_hwc2chw, module, "cuda_hwc2chw");
    if (cuRes != 0) {
        std::cerr << "cuModuleGetFunction (cuda_hwc2chw) failed with error " << cuRes << std::endl;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return -1;
    }
    CUfunction func_chw2hwc;
    cuRes = cuModuleGetFunction(&func_chw2hwc, module, "cuda_chw2hwc");
    if (cuRes != 0) {
        std::cerr << "cuModuleGetFunction (cuda_chw2hwc) failed with error " << cuRes << std::endl;
        cuModuleUnload(module);
        cuCtxDestroy(context);
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

            const size_t inputSizeBytes = pixel_size * sizeof(uint8_t);
            const size_t tempSizeBytes = pixel_size * sizeof(float);
            const size_t outputSizeBytes = pixel_size * sizeof(uint8_t);
            CUdeviceptr d_input = 0, d_temp = 0, d_output = 0;

            cuRes = cuMemAlloc(&d_input, inputSizeBytes);
            cuRes = cuMemAlloc(&d_temp, tempSizeBytes);
            cuRes = cuMemAlloc(&d_output, outputSizeBytes);

            const unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
            const unsigned int gridDimX = ((unsigned int)width + blockDimX - 1) / blockDimX;
            const unsigned int gridDimY = ((unsigned int)height + blockDimY - 1) / blockDimY;
            const unsigned int gridDimZ = 1;
            // for ready cuda kernel function(func_hwc2chw)
            size_t arg_h_val = static_cast<size_t>(height);
            size_t arg_w_val = static_cast<size_t>(width);
            size_t arg_c_val = static_cast<size_t>(channel);
            float arg_alpha_val = 1.f / 255.f;
            void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &d_input, &d_temp, &arg_alpha_val };

            auto startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                cuRes = cuLaunchKernel(
                    func_hwc2chw, gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ,
                    0, nullptr, args1, nullptr);
            }
            cuRes = cuCtxSynchronize();
            auto endTime = std::chrono::high_resolution_clock::now();
            auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            // for ready cuda kernel function(func_chw2hwc)
            size_t arg_c2_val = channel;
            size_t arg_h2_val = height;
            size_t arg_w2_val = width;
            uint8_t arg_alpha2_val = 255;
            void* args2[] = { &arg_c2_val, &arg_h2_val, &arg_w2_val, &d_temp, &d_output, &arg_alpha2_val };
            startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                cuRes = cuLaunchKernel(
                    func_chw2hwc, gridDimX, gridDimY, gridDimZ,
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

    cuModuleUnload(module);
    cuCtxDestroy(context);

    std::cout << "CUDA Benchmark completed successfully!" << std::endl;
    return 0;
}
