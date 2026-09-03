#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>

//#define SINGLE_THREAD // If you want use test the single thread, Then define it.
#include "FastChwHwcConverter.hpp"
#include "benchmark_util.hpp"

#define TEST_COUNT 10
#define WARMUP_COUNT 10
#define REPEAT_COUNT 5

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
            // Defining input and output (randomized so results are not biased
            // by all-zero data).
            const size_t pixel_size = height * width * channel;
            const std::vector<uint8_t> src_uint8 = whyb_test::random_u8(pixel_size);   // HWC input
            const std::vector<float> src_float = whyb_test::random_f32(pixel_size, 0.0f, 1.0f); // CHW input

            std::vector<float> out_float(pixel_size); // Inference output data(chw)
            std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

            const double hwc2chw_us = whyb_test::measure_min([&]() {
                whyb::cpu::hwc2chw<uint8_t, float, true>(height, width, channel, (uint8_t*)src_uint8.data(), (float*)out_float.data(), 1.0f/255.0f);
            }, WARMUP_COUNT, REPEAT_COUNT, TEST_COUNT);

            const double chw2hwc_us = whyb_test::measure_min([&]() {
                whyb::cpu::chw2hwc<float, uint8_t, true>(channel, height, width, (float*)src_float.data(), (uint8_t*)out_uint8.data(), 255.0f);
            }, WARMUP_COUNT, REPEAT_COUNT, TEST_COUNT);

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << std::fixed << std::setprecision(3)
                << hwc2chw_us / 1000.0 << "ms,\t"
                << chw2hwc_us / 1000.0 << "ms" << std::endl;
        }
    }
    std::cout << "done" << std::endl;

    return 0;
}
