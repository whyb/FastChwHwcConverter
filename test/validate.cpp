#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "FastChwHwcConverter.hpp"
#include "FastChwHwcConverterCuda.hpp"
#include "FastChwHwcConverterROCm.hpp"
#include "FastChwHwcConverterVulkan.hpp"

#include "benchmark_util.hpp"

namespace {

int g_errors = 0;

bool approx_equal(const float a, const float b) {
    // Tolerates tiny floating-point differences (e.g. FMA contraction), while
    // any real layout / formula bug produces differences many orders larger.
    const float scale = std::max({ 1.0f, std::fabs(a), std::fabs(b) });
    return std::fabs(a - b) <= 1e-4f * scale;
}

void report_error(const std::string& what, const std::string& where,
                  const size_t index, const float expected, const float actual) {
    ++g_errors;
    std::cerr << "VALIDATE FAIL: " << what << " [" << where << "] at " << index
              << " expected " << expected << " but got " << actual << std::endl;
}

std::string features_str(const char* op, const size_t c, const bool a, const bool cl, const bool nm) {
    return std::string(op) + " c=" + std::to_string(c) +
           " HasAlpha=" + (a ? "1" : "0") +
           " Clamp=" + (cl ? "1" : "0") +
           " Normalize=" + (nm ? "1" : "0");
}

// ---------------------------------------------------------------------------
// Naive scalar reference implementations (written straight from the documented
// semantics; independent from the library loops).
// ---------------------------------------------------------------------------

template <bool HasAlpha, bool NeedClamp, bool NeedNormalizedMeanStds>
void reference_hwc2chw(const size_t h, const size_t w, const size_t c,
                       const uint8_t* src, float* dst,
                       const float alpha, const float min_v, const float max_v,
                       const std::array<float, 3>& mean, const std::array<float, 3>& stds) {
    const size_t hw = h * w;
    for (size_t s = 0; s < hw; ++s) {
        const uint8_t* pixel = src + s * c;
        for (size_t ch = 0; ch < c; ++ch) {
            float value = static_cast<float>(pixel[ch]);
            if (HasAlpha) {
                value = value * alpha;
            }
            if (NeedNormalizedMeanStds) {
                // The public normalization API carries RGB parameters only.
                // RGBA reuses the final parameter set instead of reading past it.
                value = (value - mean[(std::min)(ch, size_t(2))]) / stds[(std::min)(ch, size_t(2))];
            }
            if (NeedClamp) {
                value = std::clamp(value, min_v, max_v);
            }
            dst[ch * hw + s] = value;
        }
    }
}

template <bool HasAlpha, bool NeedClamp>
void reference_chw2hwc(const size_t c, const size_t h, const size_t w,
                       const float* src, uint8_t* dst,
                       const float alpha, const float min_v, const float max_v) {
    const size_t hw = h * w;
    for (size_t s = 0; s < hw; ++s) {
        for (size_t ch = 0; ch < c; ++ch) {
            float value = src[ch * hw + s];
            if (HasAlpha) {
                value = value * alpha;
            }
            if (NeedClamp) {
                value = std::clamp(value, min_v, max_v);
            }
            dst[s * c + ch] = static_cast<uint8_t>(value);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU backend vs naive reference.
// ---------------------------------------------------------------------------

template <bool HasAlpha, bool NeedClamp, bool NeedNormalizedMeanStds>
void check_cpu_hwc2chw(const size_t h, const size_t w, const size_t c) {
    const size_t pixel_count = h * w * c;
    const auto src = whyb_test::random_u8(pixel_count);
    std::vector<float> cpu_out(pixel_count, -1.0f);
    std::vector<float> ref_out(pixel_count, -2.0f);

    const float alpha = 1.0f / 255.0f;
    const float min_v = NeedClamp ? 0.1f : 0.0f;
    const float max_v = NeedClamp ? 0.9f : 1.0f;
    const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f };
    const std::array<float, 3> stds = { 0.229f, 0.224f, 0.225f };

    whyb::cpu::hwc2chw<uint8_t, float, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(
        h, w, c, src.data(), cpu_out.data(), alpha, min_v, max_v, mean, stds);
    reference_hwc2chw<HasAlpha, NeedClamp, NeedNormalizedMeanStds>(
        h, w, c, src.data(), ref_out.data(), alpha, min_v, max_v, mean, stds);

    const std::string what = features_str("cpu.hwc2chw", c, HasAlpha, NeedClamp, NeedNormalizedMeanStds);
    for (size_t i = 0; i < pixel_count; ++i) {
        if (!approx_equal(cpu_out[i], ref_out[i])) {
            report_error(what, "vs reference", i, ref_out[i], cpu_out[i]);
        }
    }
}

template <bool HasAlpha, bool NeedClamp>
void check_cpu_chw2hwc(const size_t h, const size_t w, const size_t c) {
    const size_t pixel_count = h * w * c;
    // Without clamping the scaled value must stay inside [0,255] to keep the
    // uint8_t conversion defined, so the input stays in [0,1]. With clamping a
    // wider range is used to exercise both clamping boundaries.
    const float lo = NeedClamp ? -0.5f : 0.0f;
    const float hi = NeedClamp ? 1.5f : 1.0f;
    const auto src = whyb_test::random_f32(pixel_count, lo, hi);
    std::vector<uint8_t> cpu_out(pixel_count, 0);
    std::vector<uint8_t> ref_out(pixel_count, 0);

    const float alpha = 255.0f;
    const float min_v = 0.0f;
    const float max_v = 255.0f;
    whyb::cpu::chw2hwc<float, uint8_t, HasAlpha, NeedClamp>(
        c, h, w, src.data(), cpu_out.data(),
        static_cast<uint8_t>(alpha), static_cast<uint8_t>(min_v), static_cast<uint8_t>(max_v));
    reference_chw2hwc<HasAlpha, NeedClamp>(c, h, w, src.data(), ref_out.data(), alpha, min_v, max_v);

    const std::string what = features_str("cpu.chw2hwc", c, HasAlpha, NeedClamp, false);
    for (size_t i = 0; i < pixel_count; ++i) {
        if (cpu_out[i] != ref_out[i]) {
            report_error(what, "vs reference", i, static_cast<float>(ref_out[i]),
                         static_cast<float>(cpu_out[i]));
        }
    }
}

void validate_cpu() {
    // The last size exceeds the CPU backend parallel threshold so threaded
    // dispatch is also checked against the reference implementation.
    const size_t sizes[][2] = { { 3, 5 }, { 16, 16 }, { 33, 41 }, { 128, 512 } };
    for (size_t si = 0; si < 4; ++si) {
        const size_t h = sizes[si][0];
        const size_t w = sizes[si][1];
        for (const size_t c : { size_t(1), size_t(3), size_t(4) }) {
            check_cpu_hwc2chw<false, false, false>(h, w, c);
            check_cpu_hwc2chw<true, false, false>(h, w, c);
            check_cpu_hwc2chw<false, true, false>(h, w, c);
            check_cpu_hwc2chw<true, true, false>(h, w, c);
            check_cpu_hwc2chw<false, false, true>(h, w, c);
            check_cpu_hwc2chw<true, false, true>(h, w, c);
            check_cpu_hwc2chw<false, true, true>(h, w, c);
            check_cpu_hwc2chw<true, true, true>(h, w, c);

            check_cpu_chw2hwc<false, false>(h, w, c);
            check_cpu_chw2hwc<true, false>(h, w, c);
            check_cpu_chw2hwc<false, true>(h, w, c);
            check_cpu_chw2hwc<true, true>(h, w, c);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU backends vs CPU. The GPU kernels implement a fixed transform:
//   hwc2chw : dst = (float)src * alpha             (CHW output, no clamping)
//   chw2hwc : dst = clamp(src * alpha, 0, 255)     (HWC output)
// which corresponds to the CPU templates used in the comparison below.
// ---------------------------------------------------------------------------

struct GpuBackend {
    const char* name;
    bool (*init_fn)();
    bool (*release_fn)();
    void (*hwc2chw_fn)(size_t, size_t, size_t, const uint8_t*, float*, float);
    void (*chw2hwc_fn)(size_t, size_t, size_t, const float*, uint8_t*, uint8_t);
};

void validate_gpu_backend(const GpuBackend& gpu) {
    if (!gpu.init_fn()) {
        std::cout << "validate: " << gpu.name << " not available, skipped." << std::endl;
        return;
    }

    const size_t sizes[][2] = { { 3, 5 }, { 16, 17 }, { 64, 65 } };
    for (size_t si = 0; si < 3; ++si) {
        const size_t h = sizes[si][0];
        const size_t w = sizes[si][1];
        for (const size_t c : { size_t(1), size_t(3), size_t(4) }) {
            const size_t pixel_count = h * w * c;

            // hwc2chw (uint8 -> float) == cpu::hwc2chw<uint8_t,float,true>
            {
                const auto src = whyb_test::random_u8(pixel_count);
                std::vector<float> cpu_out(pixel_count, -1.0f);
                std::vector<float> gpu_out(pixel_count, -2.0f);
                const float alpha = 1.0f / 255.0f;
                whyb::cpu::hwc2chw<uint8_t, float, true, false, false>(
                    h, w, c, src.data(), cpu_out.data(), alpha);
                gpu.hwc2chw_fn(h, w, c, src.data(), gpu_out.data(), alpha);
                const std::string what = std::string(gpu.name) + ".hwc2chw c=" + std::to_string(c);
                for (size_t i = 0; i < pixel_count; ++i) {
                    if (!approx_equal(cpu_out[i], gpu_out[i])) {
                        report_error(what, "vs CPU", i, cpu_out[i], gpu_out[i]);
                    }
                }
            }

            // chw2hwc (float -> uint8) == cpu::chw2hwc<float,uint8_t,true,true>
            {
                const auto src = whyb_test::random_f32(pixel_count, -0.5f, 1.5f);
                std::vector<uint8_t> cpu_out(pixel_count, 0);
                std::vector<uint8_t> gpu_out(pixel_count, 0);
                const uint8_t alpha = 255;
                whyb::cpu::chw2hwc<float, uint8_t, true, true>(
                    c, h, w, src.data(), cpu_out.data(), alpha, 0, 255);
                gpu.chw2hwc_fn(c, h, w, src.data(), gpu_out.data(), alpha);
                const std::string what = std::string(gpu.name) + ".chw2hwc c=" + std::to_string(c);
                for (size_t i = 0; i < pixel_count; ++i) {
                    if (cpu_out[i] != gpu_out[i]) {
                        report_error(what, "vs CPU", i, static_cast<float>(cpu_out[i]),
                                     static_cast<float>(gpu_out[i]));
                    }
                }
            }
        }
    }

    gpu.release_fn();
    std::cout << "validate: " << gpu.name << " passed." << std::endl;
}

void validate_gpus() {
    validate_gpu_backend({ "cuda",
                           &whyb::nvidia::init, &whyb::nvidia::release,
                           [](size_t h, size_t w, size_t c, const uint8_t* s, float* d, float a) {
                               whyb::nvidia::hwc2chw(h, w, c, s, d, a);
                           },
                           [](size_t c, size_t h, size_t w, const float* s, uint8_t* d, uint8_t a) {
                               whyb::nvidia::chw2hwc(c, h, w, s, d, a);
                           } });

    validate_gpu_backend({ "rocm",
                           &whyb::amd::init, &whyb::amd::release,
                           [](size_t h, size_t w, size_t c, const uint8_t* s, float* d, float a) {
                               whyb::amd::hwc2chw(h, w, c, s, d, a);
                           },
                           [](size_t c, size_t h, size_t w, const float* s, uint8_t* d, uint8_t a) {
                               whyb::amd::chw2hwc(c, h, w, s, d, a);
                           } });

    validate_gpu_backend({ "vulkan",
                           &whyb::vulkan::init, &whyb::vulkan::release,
                           [](size_t h, size_t w, size_t c, const uint8_t* s, float* d, float a) {
                               whyb::vulkan::hwc2chw(h, w, c, s, d, a);
                           },
                           [](size_t c, size_t h, size_t w, const float* s, uint8_t* d, uint8_t a) {
                               whyb::vulkan::chw2hwc(c, h, w, s, d, a);
                           } });
}

} // namespace

int main() {
    std::cout << "validate: CPU vs naive reference ..." << std::endl;
    validate_cpu();

    std::cout << "validate: GPU (CUDA / ROCm / Vulkan) vs CPU ..." << std::endl;
    validate_gpus();

    if (g_errors != 0) {
        std::cerr << "VALIDATE FAILED with " << g_errors << " mismatch(es)." << std::endl;
        return 1;
    }
    std::cout << "All validation checks passed." << std::endl;
    return 0;
}
