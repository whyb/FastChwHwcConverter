#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>

#undef _OPENMP // If you do not want use OpenMP, Then undef it.
//#define SINGLE_THREAD // If you want use test the single thread, Then define it.
#include "FastChwHwcConverter.hpp"

#define TEST_COUNT 10

int main() {
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
            std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
            std::vector<float> src_float(pixel_size); // Source data(chw)

            std::vector<float> out_float(pixel_size); // Inference output data(chw)
            std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

            auto startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                whyb::cpu::hwc2chw<uint8_t, float>(height, width, channel, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.0f/255.0f);
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                whyb::cpu::chw2hwc<float, uint8_t>(channel, height, width, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.0f);
            }
            endTime = std::chrono::high_resolution_clock::now();
            auto chw2hwcDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime) / double(TEST_COUNT);

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << std::fixed << std::setprecision(3)
                << hwc2chwDuration.count() / 1000.0 << "ms,\t"
                << chw2hwcDuration.count() / 1000.0 << "ms" << std::endl;
        }
    }
    std::cout << "done" << std::endl;
    
    return 0;
}