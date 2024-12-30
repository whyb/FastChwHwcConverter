#include "FastChwHwcConverter.hpp"

int main() {
    const size_t ch = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output data
    uint8_t src_uint8[h * w * ch]; // Source data(hwc)
    float src_float[h * w * ch]; // Source data(chw)

    float out_float[ch * w * h]; // Inference output data(chw)
    uint8_t out_uint8[ch * w * h]; // Inference output data(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::hwc2chw<uint8_t, float>(ch, w, h, src_uint8, src_float);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::chw2hwc<float, uint8_t>(ch, w, h, out_float, out_uint8);

    return 0;
}