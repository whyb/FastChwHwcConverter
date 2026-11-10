#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "FastChwHwcConverter.hpp"
#include "FastChwHwcConverterCuda.hpp"
#include "FastChwHwcConverterOpenCL.hpp"
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
// GPU backends vs CPU. The GPU kernels implement the same transform chain as
// the CPU templates:
//   hwc2chw : dst = clamp((src * alpha - mean) / stds)   (CHW output)
//   chw2hwc : dst = clamp(src * alpha)                   (HWC output)
//
// The GPU kernels cover the uint8_t and float element types; every other
// Stype/Dtype combination routes through the automatic CPU fallback, so this
// comparison also validates that dispatch. The sweep below runs the full
// Stype x Dtype x HasAlpha x NeedClamp( x NeedNormalizedMeanStds) permutation
// matrix over the basic data types, with a value profile that keeps every
// intermediate representable for every Dtype (see check_gpu_hwc2chw).
// ---------------------------------------------------------------------------

template <typename... Types>
struct type_list {};

using AllElementTypes = type_list<bool, int8_t, uint8_t, int16_t, uint16_t,
                                 int32_t, uint32_t, int64_t, uint64_t, float, double>;

// Human readable type tag for failure messages.
template <typename T>
std::string type_tag() {
    if constexpr (std::is_same<T, bool>::value) { return "bool"; }
    else if constexpr (std::is_same<T, int8_t>::value) { return "int8"; }
    else if constexpr (std::is_same<T, uint8_t>::value) { return "uint8"; }
    else if constexpr (std::is_same<T, int16_t>::value) { return "int16"; }
    else if constexpr (std::is_same<T, uint16_t>::value) { return "uint16"; }
    else if constexpr (std::is_same<T, int32_t>::value) { return "int32"; }
    else if constexpr (std::is_same<T, uint32_t>::value) { return "uint32"; }
    else if constexpr (std::is_same<T, int64_t>::value) { return "int64"; }
    else if constexpr (std::is_same<T, uint64_t>::value) { return "uint64"; }
    else if constexpr (std::is_same<T, float>::value) { return "float"; }
    else if constexpr (std::is_same<T, double>::value) { return "double"; }
    else { return "unknown"; }
}

// Fills count elements with random values whose real value lies in [lo, hi].
template <typename T>
void fill_random(T* dst, const size_t count, const float lo, const float hi) {
    const auto values = whyb_test::random_f32(count, lo, hi);
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same<T, bool>::value) {
            dst[i] = values[i] >= 0.5f;
        }
        else {
            dst[i] = static_cast<T>(values[i]);
        }
    }
}

// float/double outputs tolerate tiny FP differences (e.g. FMA contraction on
// the normalization path). Integer outputs must match exactly, except for the
// normalization path where one float-rounding truncation flip is tolerated.
template <typename Dtype, bool FuzzyNarrow>
void compare_outputs(const std::string& what,
                     const Dtype* expected, const Dtype* actual, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_floating_point<Dtype>::value) {
            const double ev = static_cast<double>(expected[i]);
            const double av = static_cast<double>(actual[i]);
            const double scale = (std::max)({ 1.0, std::fabs(ev), std::fabs(av) });
            if (std::fabs(ev - av) > 1e-4 * scale) {
                report_error(what, "vs CPU", i, static_cast<float>(ev), static_cast<float>(av));
            }
        }
        else {
            const long long ev = static_cast<long long>(expected[i]);
            const long long av = static_cast<long long>(actual[i]);
            const long long diff = (ev > av) ? (ev - av) : (av - ev);
            if (diff > (FuzzyNarrow ? 1LL : 0LL)) {
                report_error(what, "vs CPU", i, static_cast<float>(ev), static_cast<float>(av));
            }
        }
    }
}

template <typename Stype, typename Dtype>
std::string gpu_check_name(const char* name, const char* op, const size_t c,
                           const bool a, const bool cl, const bool nm) {
    return features_str((std::string(name) + "." + op + "<" + type_tag<Stype>() +
                        "," + type_tag<Dtype>() + ">").c_str(), c, a, cl, nm);
}

template <typename Backend, typename Stype, typename Dtype,
          bool HasAlpha, bool NeedClamp, bool NeedNormalizedMeanStds>
void run_gpu_hwc2chw(const char* name, const size_t h, const size_t w, const size_t c,
                     const Dtype alpha, const Dtype min_v, const Dtype max_v,
                     const std::array<float, 3>& mean, const std::array<float, 3>& stds,
                     const float src_lo, const float src_hi) {
    const size_t pixel_count = h * w * c;
    std::unique_ptr<Stype[]> src(new Stype[pixel_count]);
    std::unique_ptr<Dtype[]> cpu_out(new Dtype[pixel_count]);
    std::unique_ptr<Dtype[]> gpu_out(new Dtype[pixel_count]);
    fill_random<Stype>(src.get(), pixel_count, src_lo, src_hi);

    whyb::cpu::hwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(
        h, w, c, src.get(), cpu_out.get(), alpha, min_v, max_v, mean, stds);
    Backend::template hwc2chw<Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(
        h, w, c, src.get(), gpu_out.get(), alpha, min_v, max_v, mean, stds);

    constexpr bool fuzzy_narrow = std::is_integral<Dtype>::value &&
                                  !std::is_same<Dtype, bool>::value &&
                                  NeedNormalizedMeanStds;
    compare_outputs<Dtype, fuzzy_narrow>(
        gpu_check_name<Stype, Dtype>(name, "hwc2chw", c, HasAlpha, NeedClamp, NeedNormalizedMeanStds),
        cpu_out.get(), gpu_out.get(), pixel_count);
}

template <typename Backend, typename Stype, typename Dtype,
          bool HasAlpha, bool NeedClamp>
void run_gpu_chw2hwc(const char* name, const size_t h, const size_t w, const size_t c,
                     const Dtype alpha, const Dtype min_v, const Dtype max_v,
                     const float src_lo, const float src_hi) {
    const size_t pixel_count = h * w * c;
    std::unique_ptr<Stype[]> src(new Stype[pixel_count]);
    std::unique_ptr<Dtype[]> cpu_out(new Dtype[pixel_count]);
    std::unique_ptr<Dtype[]> gpu_out(new Dtype[pixel_count]);
    fill_random<Stype>(src.get(), pixel_count, src_lo, src_hi);

    whyb::cpu::chw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(
        c, h, w, src.get(), cpu_out.get(), alpha, min_v, max_v);
    Backend::template chw2hwc<Stype, Dtype, HasAlpha, NeedClamp>(
        c, h, w, src.get(), gpu_out.get(), alpha, min_v, max_v);

    compare_outputs<Dtype, false>(
        gpu_check_name<Stype, Dtype>(name, "chw2hwc", c, HasAlpha, NeedClamp, false),
        cpu_out.get(), gpu_out.get(), pixel_count);
}

// Value profile for the permutation sweep: integer sources in [0, 3], other
// sources in [0, 1], alpha = 2, so every intermediate stays representable for
// every Dtype. The normalization branch may only produce negatives when the
// output type is float/double/bool or clamping protects the narrow cast; for
// the remaining combinations it runs with non-negative mean/stds instead.
template <typename Backend, typename Stype, typename Dtype,
          bool HasAlpha, bool NeedClamp, bool NeedNormalizedMeanStds>
void check_gpu_hwc2chw(const char* name, const size_t h, const size_t w, const size_t c) {
    const bool delicate_cast = NeedNormalizedMeanStds && !NeedClamp &&
        std::is_integral<Dtype>::value && !std::is_same<Dtype, bool>::value;
    const Dtype alpha = static_cast<Dtype>(2);
    const Dtype min_v = static_cast<Dtype>(0);
    const Dtype max_v = static_cast<Dtype>(3);
    const std::array<float, 3> mean = { 0.25f, 0.5f, 0.75f };
    const std::array<float, 3> stds = { 0.5f, 0.5f, 0.5f };
    const std::array<float, 3> safe_mean = { 0.0f, 0.0f, 0.0f };
    const std::array<float, 3> safe_stds = { 0.5f, 0.5f, 0.5f };
    const float src_hi = std::is_same<Stype, bool>::value ? 1.0f
                       : (std::is_integral<Stype>::value ? 3.0f : 1.0f);
    run_gpu_hwc2chw<Backend, Stype, Dtype, HasAlpha, NeedClamp, NeedNormalizedMeanStds>(
        name, h, w, c, alpha, min_v, max_v,
        delicate_cast ? safe_mean : mean, delicate_cast ? safe_stds : stds,
        0.0f, src_hi);
}

template <typename Backend, typename Stype, typename Dtype,
          bool HasAlpha, bool NeedClamp>
void check_gpu_chw2hwc(const char* name, const size_t h, const size_t w, const size_t c) {
    const Dtype alpha = static_cast<Dtype>(2);
    const Dtype min_v = static_cast<Dtype>(0);
    const Dtype max_v = static_cast<Dtype>(3);
    const float src_hi = std::is_same<Stype, bool>::value ? 1.0f
                       : (std::is_integral<Stype>::value ? 3.0f : 1.0f);
    run_gpu_chw2hwc<Backend, Stype, Dtype, HasAlpha, NeedClamp>(
        name, h, w, c, alpha, min_v, max_v, 0.0f, src_hi);
}

// All HasAlpha / NeedClamp / NeedNormalizedMeanStds combinations.
template <typename Backend, typename Stype, typename Dtype>
void sweep_gpu_hwc2chw_flags(const char* name, const size_t h, const size_t w, const size_t c) {
    check_gpu_hwc2chw<Backend, Stype, Dtype, false, false, false>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, true, false, false>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, false, true, false>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, false, false, true>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, true, true, false>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, true, false, true>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, false, true, true>(name, h, w, c);
    check_gpu_hwc2chw<Backend, Stype, Dtype, true, true, true>(name, h, w, c);
}

template <typename Backend, typename Stype, typename Dtype>
void sweep_gpu_chw2hwc_flags(const char* name, const size_t h, const size_t w, const size_t c) {
    check_gpu_chw2hwc<Backend, Stype, Dtype, false, false>(name, h, w, c);
    check_gpu_chw2hwc<Backend, Stype, Dtype, true, false>(name, h, w, c);
    check_gpu_chw2hwc<Backend, Stype, Dtype, false, true>(name, h, w, c);
    check_gpu_chw2hwc<Backend, Stype, Dtype, true, true>(name, h, w, c);
}

template <typename Backend, typename Stype, typename Dtype>
void sweep_gpu_dst_type(const char* name, const size_t h, const size_t w, const size_t c) {
    sweep_gpu_hwc2chw_flags<Backend, Stype, Dtype>(name, h, w, c);
    sweep_gpu_chw2hwc_flags<Backend, Stype, Dtype>(name, h, w, c);
}

template <typename Backend, typename Stype, typename... Dtypes>
void sweep_gpu_type_pairs(const char* name, const size_t h, const size_t w, const size_t c,
                          type_list<Dtypes...>) {
    (sweep_gpu_dst_type<Backend, Stype, Dtypes>(name, h, w, c), ...);
}

template <typename Backend, typename... Stypes>
void sweep_gpu_source_types(const char* name, const size_t h, const size_t w, const size_t c,
                            type_list<Stypes...>) {
    (sweep_gpu_type_pairs<Backend, Stypes>(name, h, w, c, AllElementTypes{}), ...);
}

// Non-default alpha / min_v / max_v / mean / stds values.
template <typename Backend>
void check_gpu_custom_parameters(const char* name, const size_t h, const size_t w, const size_t c) {
    run_gpu_hwc2chw<Backend, uint8_t, float, true, true, true>(
        name, h, w, c, 1.0f / 255.0f, -1.0f, 1.0f,
        { 0.5f, 0.4f, 0.3f }, { 0.5f, 0.25f, 0.1f }, 0.0f, 255.0f);
    run_gpu_chw2hwc<Backend, float, uint8_t, true, true>(
        name, h, w, c, uint8_t(255), uint8_t(0), uint8_t(255), -0.5f, 1.5f);
    run_gpu_chw2hwc<Backend, float, uint8_t, false, true>(
        name, h, w, c, uint8_t(1), uint8_t(16), uint8_t(235), -0.5f, 1.5f);
    run_gpu_chw2hwc<Backend, float, float, true, false>(
        name, h, w, c, 2.0f, 0.0f, 255.0f, -0.5f, 1.5f);
}

template <typename Backend>
void validate_gpu_backend(const char* name) {
    if (!Backend::init()) {
        std::cout << "validate: " << name << " not available, skipped." << std::endl;
        return;
    }

    const size_t sizes[][2] = { { 3, 5 }, { 16, 17 }, { 64, 65 } };
    for (size_t si = 0; si < 3; ++si) {
        const size_t h = sizes[si][0];
        const size_t w = sizes[si][1];
        for (const size_t c : { size_t(1), size_t(3), size_t(4) }) {
            // Full Stype x Dtype x HasAlpha x NeedClamp( x NeedNormalizedMeanStds)
            // permutation matrix over all basic data types.
            sweep_gpu_source_types<Backend>(name, h, w, c, AllElementTypes{});
            check_gpu_custom_parameters<Backend>(name, h, w, c);
        }
    }

    Backend::release();
    std::cout << "validate: " << name << " passed." << std::endl;
}

void validate_gpus() {
    validate_gpu_backend<whyb::nvidia>("cuda");
    validate_gpu_backend<whyb::amd>("rocm");
    validate_gpu_backend<whyb::vulkan>("vulkan");
    validate_gpu_backend<whyb::opencl>("opencl");
}
} // namespace

int main() {
    std::cout << "validate: CPU vs naive reference ..." << std::endl;
    validate_cpu();

    std::cout << "validate: GPU (CUDA / ROCm / Vulkan / OpenCL) vs CPU ..." << std::endl;
    validate_gpus();

    if (g_errors != 0) {
        std::cerr << "VALIDATE FAILED with " << g_errors << " mismatch(es)." << std::endl;
        return 1;
    }
    std::cout << "All validation checks passed." << std::endl;
    return 0;
}
