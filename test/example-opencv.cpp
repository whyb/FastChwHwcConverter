#include "FastChwHwcConverter.hpp"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <iostream>

int main() {
    const size_t c = 3;
    const size_t w = 1920;
    const size_t h = 1080;

    // step 1. Defining input and output 
    const cv::Size mat_size = cv::Size(w, h);
    cv::Mat src_uint8_mat = cv::Mat::zeros(mat_size, CV_8UC3);  // Source mat(hwc)
    cv::Mat src_float_mat = cv::Mat::zeros(mat_size, CV_32FC3);  // Source mat(chw)

    cv::Mat out_float_mat = cv::Mat::zeros(mat_size, CV_32FC3); // Inference output mat(chw)
    cv::Mat out_uint8_mat = cv::Mat::zeros(mat_size, CV_8UC3);  // Inference output mat(hwc)

    // step 2. Load image data to src_uint8(8U3C)

    // step 3. Convert HWC(Height, Width, Channels) to CHW(Channels, Height, Width)
    whyb::cpu::hwc2chw<uint8_t, float>(h, w, c, (uint8_t*)src_uint8_mat.data, (float*)src_float_mat.data);

    // step 4. Do AI inference
    // input: src_float ==infer==> output: out_float

    // step 5. Convert CHW(Channels, Height, Width) to HWC(Height, Width, Channels)
    whyb::cpu::chw2hwc<float, uint8_t>(c, h, w, (float*)out_float_mat.data, (uint8_t*)out_uint8_mat.data);

    std::cout << "done" << std::endl;
    return 0;
}