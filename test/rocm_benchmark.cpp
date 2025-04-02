#include <iostream>
#include <cstdint>
#include <iomanip>
#include <chrono>

#include "FastChwHwcConverterROCm.hpp"
#define TEST_COUNT 10

int main() {
    bool init = whyb::initAll();
    if (!init) { 
        std::cout << "ROCm init error!" << std::endl;
        return 0; 
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

            // 1. host memory
            std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
            std::vector<float> src_float(pixel_size); // Source data(chw)
            
            std::vector<float> out_float(pixel_size); // Inference output data(chw)
            std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

            // 2. device memory
            //hipDeviceptr_t src_uint8 = 0;
            //hipDeviceptr_t src_float = 0;
            //hipMalloc(&src_uint8, pixel_size * sizeof(uint8_t));
            //hipMalloc(&src_float, pixel_size * sizeof(float));
            //
            //hipDeviceptr_t out_float = 0;
            //hipDeviceptr_t out_uint8 = 0;
            //hipMalloc(&out_float, pixel_size * sizeof(float));
            //hipMalloc(&out_uint8, pixel_size * sizeof(uint8_t));

            auto startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {

                // 1. host memory
                whyb::hwc2chw_rocm(height, width, channel, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

                // 2. device memory
                //whyb::hwc2chw_rocm(height, width, channel, src_uint8, src_float, 1.f / 255.f);

            }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {

                // 1. host memory
                whyb::chw2hwc_rocm(channel, height, width, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

                // 2. device memory
                //whyb::chw2hwc_rocm(channel, height, width, out_float, out_uint8, 255.f);

            }
            endTime = std::chrono::high_resolution_clock::now();
            auto chw2hwcDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            // 2. device memory
            //hipFree(src_uint8);
            //hipFree(src_float);
            //hipFree(out_float);
            //hipFree(out_uint8);

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << std::fixed << std::setprecision(3)
                << hwc2chwDuration.count() / 1000.0 << "ms,\t"
                << chw2hwcDuration.count() / 1000.0 << "ms" << std::endl;
        }
    }
    std::cout << "ROCm Benchmark completed successfully!" << std::endl;
    return 0;
}
