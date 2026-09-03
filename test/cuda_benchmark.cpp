#include <iostream>
#include <cstdint>
#include <iomanip>
#include <vector>

#include "FastChwHwcConverterCuda.hpp"
#include "benchmark_util.hpp"

using namespace whyb;

#define TEST_COUNT 10000
#define WARMUP_COUNT 1000
#define REPEAT_COUNT 3

int main() {
    if (!whyb::nvidia::init()) { return 0; }
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
            const size_t pixel_size = height * width * channel;

            // Randomized host data (uploaded to device before timing).
            const std::vector<uint8_t> host_src_u8 = whyb_test::random_u8(pixel_size);
            const std::vector<float> host_src_f32 = whyb_test::random_f32(pixel_size, 0.0f, 1.0f);

            // Device buffers
            whyb::CUdeviceptr src_uint8 = 0;   // HWC uint8 input  -> CHW float (hwc2chw dst)
            whyb::CUdeviceptr src_float = 0;   // CHW float input  -> HWC uint8 (chw2hwc dst)
            whyb::CUdeviceptr out_float = 0;
            whyb::CUdeviceptr out_uint8 = 0;
            cuMemAlloc(&src_uint8, pixel_size * sizeof(uint8_t));
            cuMemAlloc(&src_float, pixel_size * sizeof(float));
            cuMemAlloc(&out_float, pixel_size * sizeof(float));
            cuMemAlloc(&out_uint8, pixel_size * sizeof(uint8_t));

            // Fill the kernel inputs with random data so the benchmark is not
            // biased by empty (all-zero) buffers.
            cuMemcpyHtoD(src_uint8, host_src_u8.data(), pixel_size * sizeof(uint8_t));
            cuMemcpyHtoD(out_float, host_src_f32.data(), pixel_size * sizeof(float));

            const double hwc2chw_us = whyb_test::measure_min([&]() {
                whyb::nvidia::hwc2chw(height, width, channel, src_uint8, src_float, 1.f / 255.f);
            }, WARMUP_COUNT, REPEAT_COUNT, TEST_COUNT);

            const double chw2hwc_us = whyb_test::measure_min([&]() {
                whyb::nvidia::chw2hwc(channel, height, width, out_float, out_uint8, 255.f);
            }, WARMUP_COUNT, REPEAT_COUNT, TEST_COUNT);

            cuMemFree(src_uint8);
            cuMemFree(src_float);
            cuMemFree(out_float);
            cuMemFree(out_uint8);

            std::cout << width << ",\t" << height << ",\t" << channel << ",\t"
                << std::fixed << std::setprecision(3)
                << hwc2chw_us / 1000.0 << "ms,\t"
                << chw2hwc_us / 1000.0 << "ms" << std::endl;
        }
    }
    whyb::nvidia::release();
    std::cout << "CUDA Benchmark completed successfully!" << std::endl;
    return 0;
}
