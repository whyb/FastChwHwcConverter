#include <iostream>
#include <windows.h>
#include <cstdint>
#include <iomanip>
#include <chrono>

// NVRTC 定义
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

// NVRTC 函数指针类型声明
typedef nvrtcResult(*nvrtcCreateProgram_t)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef nvrtcResult(*nvrtcCompileProgram_t)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult(*nvrtcGetPTXSize_t)(nvrtcProgram, size_t*);
typedef nvrtcResult(*nvrtcGetPTX_t)(nvrtcProgram, char*);
typedef nvrtcResult(*nvrtcDestroyProgram_t)(nvrtcProgram*);
typedef nvrtcResult(*nvrtcGetProgramLogSize_t)(nvrtcProgram, size_t*);
typedef nvrtcResult(*nvrtcGetProgramLog_t)(nvrtcProgram, char*);
typedef const char* (*nvrtcGetErrorString_t)(nvrtcResult);


// 定义 CUDA Driver API 函数指针类型
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;
// 函数指针类型
typedef CUresult(*cuInit_t)(unsigned int);
typedef CUresult(*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult(*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(*cuCtxDestroy_t)(CUcontext);
typedef CUresult(*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult(*cuModuleUnload_t)(CUmodule);
typedef CUresult(*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult(*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, CUcontext, void**, void**);
typedef CUresult(*cuCtxSynchronize_t)(void);
typedef CUresult(*cuMemAlloc_t)(CUdeviceptr*, unsigned int);
typedef CUresult(*cuMemFree_t)(CUdeviceptr);
typedef CUresult(*cuMemcpyHtoD_t)(CUdeviceptr, const void*, unsigned int);
typedef CUresult(*cuMemcpyDtoH_t)(void*, CUdeviceptr, unsigned int);

int main() {
    ///////////////////// NVRTC 编译过程 /////////////////////
    const char* nvrtc_dll = "nvrtc64_112_0.dll";
    HMODULE nvrtcLib = LoadLibrary(nvrtc_dll);
    if (!nvrtcLib) {
        std::cerr << "load NVRTC dll failed. dll: " << nvrtc_dll << std::endl;
        return -1;
    }

    // 获取 NVRTC 函数指针
    auto nvrtcCreateProgram_fun = (nvrtcCreateProgram_t)GetProcAddress(nvrtcLib, "nvrtcCreateProgram");
    auto nvrtcCompileProgram_fun = (nvrtcCompileProgram_t)GetProcAddress(nvrtcLib, "nvrtcCompileProgram");
    auto nvrtcGetPTXSize_fun = (nvrtcGetPTXSize_t)GetProcAddress(nvrtcLib, "nvrtcGetPTXSize");
    auto nvrtcGetPTX_fun = (nvrtcGetPTX_t)GetProcAddress(nvrtcLib, "nvrtcGetPTX");
    auto nvrtcDestroyProgram_fun = (nvrtcDestroyProgram_t)GetProcAddress(nvrtcLib, "nvrtcDestroyProgram");
    auto nvrtcGetProgramLogSize_fun = (nvrtcGetProgramLogSize_t)GetProcAddress(nvrtcLib, "nvrtcGetProgramLogSize");
    auto nvrtcGetProgramLog_fun = (nvrtcGetProgramLog_t)GetProcAddress(nvrtcLib, "nvrtcGetProgramLog");
    auto nvrtcGetErrorString_fun = (nvrtcGetErrorString_t)GetProcAddress(nvrtcLib, "nvrtcGetErrorString");

    if (!nvrtcCreateProgram_fun || !nvrtcCompileProgram_fun || !nvrtcGetPTXSize_fun ||
        !nvrtcGetPTX_fun || !nvrtcDestroyProgram_fun) {
        std::cerr << "Can't load NVRTC functions from: " << nvrtc_dll << std::endl;
        FreeLibrary(nvrtcLib);
        return -1;
    }

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

    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram_fun(&prog, cudaSource, "kernel.cu", 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cerr << "nvrtcCreateProgram failed: " << nvrtcGetErrorString_fun(res) << std::endl;
        FreeLibrary(nvrtcLib);
        return -1;
    }
    std::cout << "Create Program successfully." << std::endl;

    // 这里如果需要可增加编译选项，例如 "-default-device" 解决全局变量处理问题，可以传 options 数组
    const char* options[] = { "-default-device", "--std=c++11" };
    res = nvrtcCompileProgram_fun(prog, 2, options);
    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize_fun(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog_fun(prog, log);
        std::cerr << "CUDA Compile error: " << log << std::endl;
        delete[] log;
        nvrtcDestroyProgram_fun(&prog);
        FreeLibrary(nvrtcLib);
        return -1;
    }
    std::cout << "Compile CUDA successfully." << std::endl;

    size_t ptxSize;
    nvrtcGetPTXSize_fun(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX_fun(prog, ptx);

    // 销毁 NVRTC 程序对象
    nvrtcDestroyProgram_fun(&prog);

    //================ 使用 CUDA Driver API 加载 PTX 模块并调用内核 =================
    const char* driver_dll = "nvcuda.dll";
    HMODULE driverLib = LoadLibrary(driver_dll);
    if (!driverLib) {
        std::cerr << "Load CUDA driver dll failed: " << driver_dll << std::endl;
        delete[] ptx;
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
        delete[] ptx;
        FreeLibrary(driverLib);
        return -1;
    }

    CUresult cuRes = cuInit(0);
    if (cuRes != 0) {
        std::cerr << "cuInit failed with error " << cuRes << std::endl;
        delete[] ptx;
        FreeLibrary(driverLib);
        return -1;
    }
    CUdevice device;
    cuRes = cuDeviceGet(&device, 0);
    if (cuRes != 0) {
        std::cerr << "cuDeviceGet failed with error " << cuRes << std::endl;
        delete[] ptx;
        FreeLibrary(driverLib);
        return -1;
    }
    CUcontext context;
    cuRes = cuCtxCreate(&context, 0, device);
    if (cuRes != 0) {
        std::cerr << "cuCtxCreate failed with error " << cuRes << std::endl;
        delete[] ptx;
        FreeLibrary(driverLib);
        return -1;
    }

    // 加载编译好的 PTX 模块到 GPU 内存中
    CUmodule module;
    cuRes = cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    delete[] ptx; // 释放 PTX 缓冲区
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

    // 设置图像参数（1024x1024，3 通道，HWC 排布）
    const int width_img = 7680;
    const int height_img = 4320;
    const unsigned int channels = 4;
    const size_t numPixels = width_img * height_img;
    const size_t inputSizeBytes = numPixels * channels * sizeof(uint8_t);
    const size_t tempSizeBytes = numPixels * channels * sizeof(float);
    const size_t outputSizeBytes = numPixels * channels * sizeof(uint8_t);

    CUdeviceptr d_input = 0, d_temp = 0, d_output = 0;
    cuRes = cuMemAlloc(&d_input, inputSizeBytes);
    if (cuRes != 0) {
        std::cerr << "cuMemAlloc (d_input) failed with error " << cuRes << std::endl;
        goto cleanup;
    }
    cuRes = cuMemAlloc(&d_temp, tempSizeBytes);
    if (cuRes != 0) {
        std::cerr << "cuMemAlloc (d_temp) failed with error " << cuRes << std::endl;
        goto cleanup_dinput;
    }
    cuRes = cuMemAlloc(&d_output, outputSizeBytes);
    if (cuRes != 0) {
        std::cerr << "cuMemAlloc (d_output) failed with error " << cuRes << std::endl;
        goto cleanup_dtemp;
    }

    // 初始化主机输入数据（HWC 排布，uint8_t 类型）
    uint8_t* host_input = new uint8_t[inputSizeBytes];
    for (size_t i = 0; i < inputSizeBytes; i++) {
        host_input[i] = static_cast<uint8_t>(i % 256);
    }
    cuRes = cuMemcpyHtoD(d_input, host_input, inputSizeBytes);
    delete[] host_input;
    if (cuRes != 0) {
        std::cerr << "cuMemcpyHtoD (d_input) failed with error " << cuRes << std::endl;
        goto cleanup_doutput;
    }

    // 设置 kernel 执行参数
    unsigned int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    unsigned int gridDimX = (width_img + blockDimX - 1) / blockDimX;
    unsigned int gridDimY = (height_img + blockDimY - 1) / blockDimY;
    unsigned int gridDimZ = 1;

    // 第一阶段：执行 cuda_hwc2chw 内核
    size_t arg_h_val = static_cast<size_t>(height_img);
    size_t arg_w_val = static_cast<size_t>(width_img);
    size_t arg_c_val = static_cast<size_t>(channels);
    float arg_alpha_val = 1.f / 255.f;
    void* args1[] = { &arg_h_val, &arg_w_val, &arg_c_val, &d_input, &d_temp, &arg_alpha_val };

    auto startTime = std::chrono::high_resolution_clock::now();
    cuRes = cuLaunchKernel(func_hwc2chw, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args1, nullptr);
    if (cuRes != 0) {
        std::cerr << "cuLaunchKernel (cuda_hwc2chw) failed with error " << cuRes << std::endl;
        goto cleanup_doutput;
    }
    cuRes = cuCtxSynchronize();
    if (cuRes != 0) {
        std::cerr << "cuCtxSynchronize after cuda_hwc2chw failed with error " << cuRes << std::endl;
        goto cleanup_doutput;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "hwc2chw, " << width_img << ",\t" << height_img << ",\t" << channels << ",\t"
        << std::fixed << std::setprecision(3)
        << hwc2chwDuration.count() / 1000.0 << "ms" << std::endl;

    // 第二阶段：执行 cuda_chw2hwc 内核
    size_t arg_c2_val = static_cast<size_t>(channels);
    size_t arg_h2_val = static_cast<size_t>(height_img);
    size_t arg_w2_val = static_cast<size_t>(width_img);
    uint8_t arg_alpha2_val = 255;
    void* args2[] = { &arg_c2_val, &arg_h2_val, &arg_w2_val, &d_temp, &d_output, &arg_alpha2_val };

    startTime = std::chrono::high_resolution_clock::now();
    cuRes = cuLaunchKernel(func_chw2hwc, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        0, nullptr, args2, nullptr);
    if (cuRes != 0) {
        std::cerr << "cuLaunchKernel (cuda_chw2hwc) failed with error " << cuRes << std::endl;
        goto cleanup_doutput;
    }
    cuRes = cuCtxSynchronize();
    if (cuRes != 0) {
        std::cerr << "cuCtxSynchronize after cuda_chw2hwc failed with error " << cuRes << std::endl;
        goto cleanup_doutput;
    }
    endTime = std::chrono::high_resolution_clock::now();
    auto chw2hwcDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "chw2hwc, " << width_img << ",\t" << height_img << ",\t" << channels << ",\t"
        << std::fixed << std::setprecision(3)
        << chw2hwcDuration.count() / 1000.0 << "ms" << std::endl;

    // 将结果从设备拷贝回主机
    uint8_t* host_output = new uint8_t[outputSizeBytes];
    cuRes = cuMemcpyDtoH(host_output, d_output, outputSizeBytes);
    if (cuRes != 0) {
        std::cerr << "cuMemcpyDtoH failed with error " << cuRes << std::endl;
        delete[] host_output;
        goto cleanup_doutput;
    }

    // 输出部分结果，例如前 100 个像素的值
    // std::cout << "Output data (first 100 values):" << std::endl;
    // for (size_t i = 0; i < 100 && i < outputSizeBytes; i++) {
    //     std::cout << static_cast<int>(host_output[i]) << " ";
    // }
    // std::cout << std::endl;
    delete[] host_output;


cleanup_doutput:
    cuMemFree(d_output);
cleanup_dtemp:
    cuMemFree(d_temp);
cleanup_dinput:
    cuMemFree(d_input);
cleanup:
    cuModuleUnload(module);
    cuCtxDestroy(context);
    FreeLibrary(driverLib);

    std::cout << "CUDA Benchmark completed successfully!" << std::endl;
    return 0;
}
