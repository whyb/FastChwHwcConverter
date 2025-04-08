#include "FastChwHwcConverter.hpp"
#include "FastChwHwcConverterCuda.hpp"
#include "FastChwHwcConverterROCm.hpp"
#include <vector>
#include <cstdint>
#include <iostream>

void cpu_example()
{
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::cpu::hwc2chw<uint8_t, float>(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::cpu::chw2hwc<float, uint8_t>(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    std::cout << "cpu example done" << std::endl;
}

void cuda_example()
{
    if (!whyb::nvidia::init()) { return; }
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::nvidia::hwc2chw(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f/255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::nvidia::chw2hwc(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    whyb::nvidia::release();
    std::cout << "cuda example done" << std::endl;
}

void rocm_example()
{
    if (!whyb::amd::init()) { return; }
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const size_t pixel_size = h * w * c;
    std::vector<uint8_t> src_uint8(pixel_size); // Source data(hwc)
    std::vector<float> src_float(pixel_size); // Source data(chw)

    std::vector<float> out_float(pixel_size); // Inference output data(chw)
    std::vector<uint8_t> out_uint8(pixel_size); // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::amd::hwc2chw(h, w, c, (uint8_t*)src_uint8.data(), (float*)src_float.data(), 1.f / 255.f);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::amd::chw2hwc(c, h, w, (float*)out_float.data(), (uint8_t*)out_uint8.data(), 255.f);

    whyb::amd::release();
    std::cout << "rocm example done" << std::endl;
}

int main() {
    cpu_example();
    cuda_example();
    rocm_example();
    return 0;
}