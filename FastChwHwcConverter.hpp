/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2024-2025] [張小凡](https://github.com/whyb)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Check if the compiler supports C++17
#if __cplusplus >= 201703L
// If C++17 is supported, use std::clamp from the standard library
#define STD_CLAMP(value, low, high) (std::clamp)(value, low, high)
#else
// If C++17 is not supported but C++11 is, implement std::clamp using std::min and std::max
#define STD_CLAMP(value, low, high) ((std::max)(low, (std::min)(value, high)))
#endif

namespace whyb {
    class cpu {
    private:
        cpu() {
            static bool init0([]() {
                return true;
                }());
        }
    public:
        ~cpu() = default;
        cpu(const cpu&) = delete;
        cpu& operator=(const cpu&) = delete;
        cpu(cpu&&) = delete;
        cpu& operator=(cpu&&) = delete;

    public:
        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        * @param clamp Whether to clamp the data values
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        * @param normalized_mean_stds Whether to use mean and standard deviation for normalization
        * @param mean Array of mean values for normalization
        * @param stds Array of standard deviation values for normalization
        */
        template <typename Stype, typename Dtype>
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const bool clamp = false, const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const bool normalized_mean_stds = false,
            const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
            const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }) {

            if constexpr (std::is_same_v<Stype, uint8_t> &&
                std::is_same_v<Dtype, float>)
            {
                if (is_avx2_supported()) {
                    if (clamp) {
                        return normalized_mean_stds ?
                            hwc2chw_avx2_impl<true, true, false>(
                                h, w, c, src, dst, alpha, min_v, max_v, mean, stds) :
                            hwc2chw_avx2_impl<true, false, false>(
                                h, w, c, src, dst, alpha, min_v, max_v, mean, stds);
                    }
                    else {
                        return normalized_mean_stds ?
                            hwc2chw_avx2_impl<false, true, false>(
                                h, w, c, src, dst, alpha, min_v, max_v, mean, stds) :
                            hwc2chw_avx2_impl<false, false, false>(
                                h, w, c, src, dst, alpha, min_v, max_v, mean, stds);
                    }
                }
                // TODO else : SSE4.1 / SSE2 impl ...  e.g. _mm_cvtepu8_epi32
            }

            std::function<Dtype(const Stype&, const size_t&)> cvt_fun;
            if (clamp) {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>((src_val - mean[c]) / stds[c], min_v, max_v)); };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val, min_v, max_v)); };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>((src_val * alpha - mean[c]) / stds[c], min_v, max_v)); };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val * alpha, min_v, max_v)); };
                    }
                }
            }
            else {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>((src_val - mean[c]) / stds[c]); };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val); };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>((src_val * alpha - mean[c]) / stds[c]); };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val * alpha); };
                    }
                }
            }

#ifdef _OPENMP
            const size_t hw_stride = w * h;
            const size_t num_threads = omp_get_max_threads();
            const size_t chunk_size = hw_stride / num_threads;
#pragma omp parallel
            {
                const size_t thread_id = omp_get_thread_num();
                const size_t start_idx = thread_id * chunk_size;
                const size_t end_idx = (thread_id == num_threads - 1) ? hw_stride : (start_idx + chunk_size);
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[stride_index] = cvt_fun(src[index++], c1);
                    }
                }
            }
#else
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[stride_index] = cvt_fun(src[index++], c1);
                }
            }
#endif
        }


        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        * @param clamp Whether to clamp the data values
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        */
        template <typename Stype, typename Dtype>
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const bool clamp = false, const Dtype min_v = 0, const Dtype max_v = 255) {
            std::function<Dtype(const Stype&, const size_t&)> cvt_fun;
            if (clamp) {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val * alpha, min_v, max_v)); };
                }
                else {
                    cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val, min_v, max_v)); };
                }
            }
            else {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val); };
                }
                else {
                    cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val * alpha); };
                }
            }

#ifdef _OPENMP
            const size_t hw_stride = w * h;
            const size_t num_threads = omp_get_max_threads();
            const size_t chunk_size = hw_stride / num_threads;
#pragma omp parallel
            {
                const size_t thread_id = omp_get_thread_num();
                const size_t start_idx = thread_id * chunk_size;
                const size_t end_idx = (thread_id == num_threads - 1) ? hw_stride : (start_idx + chunk_size);
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[index++] = cvt_fun(src[stride_index], c1);
                    }
                }
            }
#else
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[index++] = cvt_fun(src[stride_index], c1);
                }
            }
#endif
        }

    private:
        template <typename T>
        inline static T std_clamp(const T& value, const T& low, const T& high) {
            return STD_CLAMP(value, low, high);
        }

        /**
        * @brief Determines if two numbers are approximately equal
        *
        * @tparam Type Type of the numbers
        * @param a First number
        * @param b Second number
        * @return true if the numbers are approximately equal
        * @return false if the numbers are not approximately equal
        */
        template <typename Type>
        inline static bool is_number_equal(const Type& a, const Type& b) {
            static Type epsilon = std::numeric_limits<Type>::epsilon();
            return std::abs(a - b) < epsilon;
        }

        static bool is_avx2_supported() {
#if defined(__AVX2__)
            return true;
#else
            int regs[4] = { 0 };

#if defined(_MSC_VER)
            __cpuidex(regs, 1, 0);
#elif defined(__GNUC__) || defined(__clang__)
            __cpuid_count(1, 0, regs[0], regs[1], regs[2], regs[3]);
#endif
            const bool osxsave = (regs[2] & (1 << 27)) != 0;

#if defined(_MSC_VER)
            __cpuidex(regs, 7, 0);
#elif defined(__GNUC__) || defined(__clang__)
            __cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
#endif
            const bool avx2 = (regs[1] & (1 << 5)) != 0;

            if (osxsave) {
                const uint64_t xcr0 = _xgetbv(0);
                return (xcr0 & 0x6) == 0x6 && avx2;
            }
            return false;
#endif
        }

        // AVX2  (uint8_t -> float)
        template <bool Clamp, bool Normalized, bool AlphaIsOne>
        static void hwc2chw_avx2_impl(
            const size_t h, const size_t w, const size_t c,
            const uint8_t* src, float* dst,
            const float alpha,
            const float min_v, const float max_v,
            const std::array<float, 3>& mean, const std::array<float, 3>& stds)
        {
            constexpr size_t simd_elements = 32;
            const size_t hw = h * w;
            const size_t total_pixels = hw * c;
            const size_t blocks = total_pixels / simd_elements;
            const size_t remainder = total_pixels % simd_elements;

            const __m256 v_alpha = _mm256_set1_ps(alpha);
            const __m256 v_min = _mm256_set1_ps(min_v);
            const __m256 v_max = _mm256_set1_ps(max_v);
            const __m256 v_mean[3] = {
                _mm256_set1_ps(mean[0]),
                _mm256_set1_ps(mean[1]),
                _mm256_set1_ps(mean[2])
            };
            const __m256 v_std[3] = {
                _mm256_set1_ps(stds[0]),
                _mm256_set1_ps(stds[1]),
                _mm256_set1_ps(stds[2])
            };

            const uint8_t* src_ptr = src;
            float* dst_ptr = dst;

            for (size_t i = 0; i < blocks; ++i) {
                __m256i u8_data = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(src_ptr));

                // 分4次处理32字节
                for (int batch = 0; batch < 4; ++batch) {
                    __m128i u8x8;

                    // 步骤1：获取基础数据块
                    if (batch < 2) {
                        u8x8 = _mm256_extractf128_si256(u8_data, 0); // 低128位
                    }
                    else {
                        u8x8 = _mm256_extractf128_si256(u8_data, 1); // 高128位
                    }

                    // 步骤2：硬编码位移处理
                    switch (batch) {
                    case 1: case 3:
                        u8x8 = _mm_srli_si128(u8x8, 8); // 常量参数8
                        break;
                    default:
                        break; // 0位移
                    }

                    // 步骤3：转换到float32
                    __m256i u32x8 = _mm256_cvtepu8_epi32(u8x8);
                    __m256 f32x8 = _mm256_cvtepi32_ps(u32x8);

                    // 步骤4：Alpha缩放（条件编译保留）
                    if (!AlphaIsOne) {
                        f32x8 = _mm256_mul_ps(f32x8, v_alpha);
                    }

                    // 步骤5：归一化处理（条件编译保留）
                    if (Normalized) {
                        const size_t base_idx = i * simd_elements + batch * 8;
                        const size_t channel = base_idx % c;
                        f32x8 = _mm256_sub_ps(f32x8, v_mean[channel]);
                        f32x8 = _mm256_div_ps(f32x8, v_std[channel]);
                    }

                    // 步骤6：值裁剪（条件编译保留）
                    if (Clamp) {
                        f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, v_min), v_max);
                    }

                    // 步骤7：分高低128位存储
                    const size_t base_offset = i * simd_elements + batch * 8;

                    // 处理低128位（k=0-3）
                    __m128 low_lane = _mm256_extractf128_ps(f32x8, 0);
                    for (int k = 0; k < 4; ++k) {
                        const size_t global_idx = base_offset + k;
                        const size_t c_idx = global_idx % c;
                        const size_t hw_idx = global_idx / c;
                        const size_t dst_pos = c_idx * hw + hw_idx;

                        float val;
                        switch (k) {
                        case 0: val = _mm_cvtss_f32(low_lane); break;
                        case 1: val = _mm_cvtss_f32(_mm_movehdup_ps(low_lane)); break;
                        case 2: val = _mm_cvtss_f32(_mm_movehl_ps(low_lane, low_lane)); break;
                        case 3: val = _mm_cvtss_f32(_mm_shuffle_ps(low_lane, low_lane, _MM_SHUFFLE(3, 3, 3, 3))); break;
                        }
                        dst[dst_pos] = val;
                    }

                    // 处理高128位（k=4-7）
                    __m128 high_lane = _mm256_extractf128_ps(f32x8, 1);
                    for (int k = 4; k < 8; ++k) {
                        const size_t global_idx = base_offset + k;
                        const size_t c_idx = global_idx % c;
                        const size_t hw_idx = global_idx / c;
                        const size_t dst_pos = c_idx * hw + hw_idx;

                        float val;
                        switch (k - 4) { // 转换为0-3
                        case 0: val = _mm_cvtss_f32(high_lane); break;
                        case 1: val = _mm_cvtss_f32(_mm_movehdup_ps(high_lane)); break;
                        case 2: val = _mm_cvtss_f32(_mm_movehl_ps(high_lane, high_lane)); break;
                        case 3: val = _mm_cvtss_f32(_mm_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3))); break;
                        }
                        dst[dst_pos] = val;
                    }
                }

                src_ptr += simd_elements;
            }

            // 处理剩余元素（标量回退）
            for (size_t i = blocks * simd_elements; i < total_pixels; ++i) {
                const size_t c_idx = i % c;
                const size_t hw_idx = i / c;
                float val = static_cast<float>(src[i]);

                if (!AlphaIsOne) val *= alpha;
                if (Normalized) val = (val - mean[c_idx]) / stds[c_idx];
                if (Clamp) val = STD_CLAMP(val, min_v, max_v);

                dst[c_idx * hw + hw_idx] = val;
            }
        }
        
    };

}
