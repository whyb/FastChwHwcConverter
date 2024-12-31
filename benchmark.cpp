#include <chrono>
#include <iostream>
#include <vector>
#include <utility>

#undef _OPENMP // If you want to test single-threading. Then enable(undef) it.
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
                whyb::hwc2chw<uint8_t, float>(channel, width, height, (uint8_t*)src_uint8.data(), (float*)src_float.data());
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto hwc2chwDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime) / TEST_COUNT;

            startTime = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < TEST_COUNT; ++i) {
                whyb::chw2hwc<float, uint8_t>(channel, width, height, (float*)out_float.data(), (uint8_t*)out_uint8.data());
            }
            endTime = std::chrono::high_resolution_clock::now();
            auto chw2hwcDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime) / TEST_COUNT;

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << hwc2chwDuration.count() << "ms,\t" << chw2hwcDuration.count() << "ms" << std::endl << std::flush;
        }
    }
    
    return 0;
}